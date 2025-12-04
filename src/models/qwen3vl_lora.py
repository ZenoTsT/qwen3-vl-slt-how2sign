# src/models/qwen3vl_lora.py

import time
from typing import Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model


def load_qwen3vl_lora(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    r: int = 16,                # LoRA rank (dimensione dei low-rank adapters)
    alpha: int = 32,            # scala di LoRA (quanto "pesa" l'aggiornamento)
    dropout: float = 0.05,      # dropout nei layer LoRA
    device: Optional[str] = None,
):
    """
    Carica Qwen3-VL + processor e attacca i layer LoRA sui layer di attenzione.

    Parametri:
        - model_name:   nome del modello HuggingFace
        - r, alpha, dropout: iperparametri LoRA
        - device:       stringa tipo "cuda", "cuda:0", "cpu".
                        Se None -> inferito automaticamente.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 1) Processor (tokenizer + image processor)
    # -----------------------------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    t1 = time.perf_counter()
    print(f"[TIMER] Processor loaded in {t1 - t0:.2f} seconds.")

    # -----------------------------
    # 2) Modello base (su CPU, poi spostiamo)
    # -----------------------------
    t2 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,     # fp16 per risparmiare memoria
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    t3 = time.perf_counter()
    print(f"[TIMER] Base model loaded in {t3 - t2:.2f} seconds.")

    # -----------------------------
    # 3) LoRA su layer di attenzione
    # -----------------------------
    t4 = time.perf_counter()

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            # nomi dei sotto-moduli di attention in Qwen
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    t5 = time.perf_counter()
    print(f"[TIMER] LoRA adapters attached in {t5 - t4:.2f} seconds.")

    # -----------------------------
    # 4) Sposta modello sul device scelto
    # -----------------------------
    model.to(device)

    # -----------------------------
    # 5) Conta parametri trainabili (solo LoRA)
    # -----------------------------
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    print(f"[TIMER] Total load_qwen3vl_lora() time: {t5 - t0:.2f} seconds.")

    return model, processor