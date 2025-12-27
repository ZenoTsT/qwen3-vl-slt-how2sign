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
    Carica Qwen3-VL + processor e attacca i layer LoRA sui layer di attenzione.

    stage="stage1":
        - LoRA su Vision + Merger/Projection (LLM frozen)
    stage="stage2":
        - (opzionale) merge del LoRA di stage1 nel base model (Vision+Merger)
        - LoRA trainabile su Merger/Projection + LLM (Vision frozen)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 1) Processor
    # -----------------------------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    t1 = time.perf_counter()
    print(f"[TIMER] Processor loaded in {t1 - t0:.2f} seconds.")

    # -----------------------------
    # 2) Modello base
    # -----------------------------
    t2 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    
    if stage == "stage2":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable()

    t3 = time.perf_counter()
    print(f"[TIMER] Base model loaded in {t3 - t2:.2f} seconds.")

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

            # 1) Costruisco un PEFT model con LoRA sugli stessi target_modules di stage1
            stage1_lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=TARGET_MODULES_STAGE1,
                task_type="CAUSAL_LM",
            )
            stage1_lora_model = get_peft_model(model, stage1_lora_config)

            # 2) Carico i pesi LoRA da adapter.pt (solo chiavi "lora_")
            adapter_state = torch.load(adapter_path, map_location="cpu")

            # strict=False perché lo state_dict contiene solo i pesi LoRA
            incompat = stage1_lora_model.load_state_dict(
                adapter_state,
                strict=False,
            )

            missing = incompat.missing_keys
            unexpected = incompat.unexpected_keys

            if missing:
                print(f"[WARN] Stage1 LoRA missing keys (ok se non sono lora_): {len(missing)}")
            if unexpected:
                print(f"[WARN] Stage1 LoRA unexpected keys: {len(unexpected)}")

            # 3) Merge: i pesi LoRA di stage1 vengono fusi nel base model
            stage1_lora_model = stage1_lora_model.merge_and_unload()
            model = stage1_lora_model  # ora è di nuovo un Qwen3VLForConditionalGeneration

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