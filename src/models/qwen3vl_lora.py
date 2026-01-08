# src/models/qwen3vl_lora.py

from pathlib import Path
import time
from typing import Optional, Literal, List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model


# ------------------------------------------------------------
# Target LoRA per i due stage (basati sul dump dei moduli)
# ------------------------------------------------------------
TARGET_MODULES_STAGE1 = [
    # ---- Vision tower (blocks.*)
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    # ---- Projection / Merger
    "visual.merger.linear_fc1",
    "visual.merger.linear_fc2",
    "visual.deepstack_merger_list.0.linear_fc1",
    "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.1.linear_fc1",
    "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.2.linear_fc1",
    "visual.deepstack_merger_list.2.linear_fc2",
]

TARGET_MODULES_STAGE2 = [
    # ---- Projection / Merger (sempre trainabile anche in stage2)
    "visual.merger.linear_fc1",
    "visual.merger.linear_fc2",
    "visual.deepstack_merger_list.0.linear_fc1",
    "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.1.linear_fc1",
    "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.2.linear_fc1",
    "visual.deepstack_merger_list.2.linear_fc2",
    # ---- Language model (LLM)
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def _freeze_all_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _ensure_only_lora_trainable(model: torch.nn.Module) -> None:
    """
    Forza che SOLO i pesi LoRA siano trainabili.
    """
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def load_qwen3vl_lora(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    device: Optional[str] = None,
    stage: Literal["stage1", "stage2"] = "stage1",
    stage1_adapter_dir: str | None = None,
):
    """
    Load Qwen3-VL together with its processor and attach LoRA adapters.

    stage="stage1":
      - Train LoRA on the Vision tower + Merger/Projection (LLM is frozen)

    stage="stage2":
      - (Optional) merge the Stage1 LoRA into the base model (Vision+Merger)
      - Train LoRA on the Merger/Projection + LLM (Vision is frozen)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 1) Load the processor
    # -----------------------------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    t1 = time.perf_counter()
    print(f"[TIMER] Processor loaded in {t1 - t0:.2f} seconds.")

    # -----------------------------
    # 2) Load the base model
    # -----------------------------
    t2 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.config.use_cache = False # during inference, store key/value states of the attention (saves memory), incompatible with gradient checkpointing
    
    model.gradient_checkpointing_enable( # save less activation recalculing some forward steps during backward (more memory efficient less speed)
        gradient_checkpointing_kwargs={"use_reentrant": False} # new gradient checkpointing implementation
    )
    
    # SE FUNZIONA IN ENTRAMBI I MODI, LASCIO COSÌ
    # if stage == "stage2":
    #     model.gradient_checkpointing_enable( # save less activation recalculing some forward steps during backward (more memory efficient less speed)
    #         gradient_checkpointing_kwargs={"use_reentrant": False} # new gradient checkpointing implementation
    #     )
    # else:
    #     model.gradient_checkpointing_enable()

    t3 = time.perf_counter()
    print(f"[TIMER] Base model loaded in {t3 - t2:.2f} seconds.")

    # Freeze all parameters initially, after that we will unfreeze only LoRA
    _freeze_all_params(model)

    # -----------------------------
    # 3) (solo stage2) merge Stage1 LoRA nel base model
    # -----------------------------
    t4 = time.perf_counter()

    if stage == "stage2":
        can_load_stage1 = False
        adapter_path: Path | None = None

        if stage1_adapter_dir is not None:
            stage1_dir = Path(stage1_adapter_dir)
            adapter_path = stage1_dir / "adapter.pt"
            if adapter_path.exists():
                can_load_stage1 = True

        if can_load_stage1 and adapter_path is not None:
            print(f"[INFO] Stage2: loading Stage1 LoRA from {adapter_path}")

            # 1) I build PEFT model with LoRA on the same target_modules as stage1
            # PEFT stands for Parameter-Efficient Fine-Tuning and a PEFT model is a wrapper
            # around the base model that adds LoRA adapters (or similar techniques) to the specified target modules.
            stage1_lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=TARGET_MODULES_STAGE1,
                task_type="CAUSAL_LM",
            )
            stage1_lora_model = get_peft_model(model, stage1_lora_config)

            # 2) Load LoRA form adapter.pt (only "lora_" keys)
            adapter_state = torch.load(adapter_path, map_location="cpu")

            ckpt_lora_keys = [k for k in adapter_state.keys() if "lora_" in k]
            model_keys = set(stage1_lora_model.state_dict().keys())
            matched_lora_keys = [k for k in ckpt_lora_keys if k in model_keys]
            print("[DEBUG][Stage1->Stage2] adapter_state keys:", len(adapter_state))
            print("[DEBUG][Stage1->Stage2] adapter_state LoRA keys:", len(ckpt_lora_keys))
            print("[DEBUG][Stage1->Stage2] matched LoRA keys in model:", len(matched_lora_keys))
            print(f"[DEBUG] Match ratio: {len(matched_lora_keys)}/{len(ckpt_lora_keys)} = {len(matched_lora_keys)/max(len(ckpt_lora_keys),1):.2%}")
            lora_norm_sq = 0.0
            for n, p in stage1_lora_model.named_parameters():
                if "lora_" in n:
                    lora_norm_sq += float(p.detach().float().pow(2).sum().cpu())
            print(f"[DEBUG] Stage1 LoRA global L2 norm (pre-merge): {(lora_norm_sq**0.5):.6f}")

            # strict=False since state_dict contains only LoRA keys, non all model keys
            incompat = stage1_lora_model.load_state_dict(
                adapter_state,
                strict=False,
            )

            missing = incompat.missing_keys
            unexpected = incompat.unexpected_keys
            print(f"[DEBUG][Stage1->Stage2] load_state_dict missing_keys: {len(missing)}")
            print(f"[DEBUG][Stage1->Stage2] load_state_dict unexpected_keys: {len(unexpected)}")

            # 3) Merge: LoRA stage1 parameters are merged into base model
            stage1_lora_model = stage1_lora_model.merge_and_unload()
            model = stage1_lora_model  # now it's a new Qwen3VLForConditionalGeneration
            num_lora_params_after_merge = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
            print(f"[DEBUG][Stage1->Stage2] LoRA params after merge: {num_lora_params_after_merge}")
            if num_lora_params_after_merge != 0:
                print("[WARN][Stage1->Stage2] LoRA params still present after merge_and_unload()! Something is off.")
            print(f"[DEBUG][Stage1->Stage2] model class after merge: {model.__class__.__name__}")

            # 4) Congelo tutto: stage2 attaccherà nuovi LoRA
            _freeze_all_params(model)

            print("[INFO] Stage1 LoRA merged into base model and frozen.")
        else:
            print("[WARN] Stage2: Stage1 adapter not found or invalid -> running Stage2 standalone (no Stage1 loaded).")

    # -----------------------------
    # 4) Applico LoRA trainabile per lo stage richiesto
    # -----------------------------
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES_STAGE1 if stage == "stage1" else TARGET_MODULES_STAGE2,
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, lora_config)
    t5 = time.perf_counter()
    print(f"[TIMER] LoRA adapters attached in {t5 - t4:.2f} seconds.")
    print(f"[INFO] LoRA stage = {stage}")

    # Solo LoRA trainabile
    _ensure_only_lora_trainable(lora_model)

    # -----------------------------
    # 5) Sposto su device
    # -----------------------------
    lora_model.to(device)

    # -----------------------------
    # 6) Statistiche parametri
    # -----------------------------
    lora_modules = [
        name for name, module in lora_model.named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    ]

    print(f"[DEBUG] Number of modules with LoRA injected: {len(lora_modules)}")
    print("[DEBUG] Example LoRA-injected modules (up to 10):")
    for n in lora_modules[:10]:
        print("  ", n)
        
    trainable, total = 0, 0
    for _, p in lora_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    trainable_names = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    print(f"[DEBUG] Trainable tensors (showing up to 5):")
    for n in trainable_names[:5]:
        print("  ", n)

    print(f"[TIMER] Total load_qwen3vl_lora() time: {t5 - t0:.2f} seconds.")

    return lora_model, processor