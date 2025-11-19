# src/models/qwen3vl_lora.py

import time
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model


def load_qwen3vl_lora(                                      # ΔW ≈ A · B, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct", 
    r: int = 16,                                            # LoRA rank - dimension of the LoRA matrices
    alpha: int = 32,                                        # scale factor (how much weight to give to LoRA)
    dropout: float = 0.05,                                  # dropout used for LoRA
    device_map: str = "auto",                               # how to map the model to devices/GPUs
):
    """
    Load Qwen3-VL vision-language model with LoRA applied.
    Also prints rough timing for each main step.
    """

    # -----------------------------
    # 1) Load processor
    # -----------------------------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(              # load the multimodal processor (tokenizer + image processor, the "vocabulary" and preprocessing of the MLLM)
        model_name, 
        trust_remote_code=True,
    )
    t1 = time.perf_counter()
    print(f"[TIMER] Processor loaded in {t1 - t0:.2f} seconds.")

    # -----------------------------
    # 2) Load base model
    # -----------------------------
    t2 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(  # load the pretrained multimodal causal LM (text + vision)
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="sdpa",
        # attn_implementation="eager",
        trust_remote_code=True,
    )
    t3 = time.perf_counter()
    print(f"[TIMER] Base model loaded in {t3 - t2:.2f} seconds.")

    # -----------------------------
    # 3) Attach LoRA
    # -----------------------------
    t4 = time.perf_counter()
    # Config LoRA (for now only on attention projections, we will refine this later)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],    # we apply LoRA to queries, keys, values and outputs of attention
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)              # wrap the base model with LoRA adapters
    t5 = time.perf_counter()
    print(f"[TIMER] LoRA adapters attached in {t5 - t4:.2f} seconds.")

    # Print how many parameters are trainable
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    # Total time for the whole loading pipeline
    print(f"[TIMER] Total load_qwen3vl_lora() time: {t5 - t0:.2f} seconds.")

    return model, processor