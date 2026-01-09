# src/models/qwen3vl_lora_full.py
"""
Qwen3-VL + LoRA "full" (su tutto il modello) — script 1/2

Obiettivo:
- caricare processor + modello base
- attaccare LoRA su *tutti* i layer lineari (target_modules="all-linear")
- lasciare trainabili SOLO i pesi LoRA
- utility per caricare/salvare adapter.pt che contiene SOLO le chiavi "lora_"

Nota:
- Questo file NON fa training. È pensato per essere importato dal training script.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model


# -----------------------------
# Helpers: freeze / trainable
# -----------------------------
def _freeze_all_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _ensure_only_lora_trainable(model: torch.nn.Module) -> None:
    """
    Forza che SOLO i pesi LoRA siano trainabili.
    """
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)


def _print_trainable_stats(model: torch.nn.Module) -> None:
    trainable, total = 0, 0
    trainable_names = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_names.append(n)

    pct = (100.0 * trainable / max(total, 1))
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    print("[DEBUG] Trainable tensors (showing up to 10):")
    for n in trainable_names[:10]:
        print("  ", n)


# -----------------------------
# LoRA attach (ALL LINEAR)
# -----------------------------
def attach_lora_all_linear(
    model: torch.nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> torch.nn.Module:
    """
    Attacca LoRA su tutti i moduli lineari supportati da PEFT:
    target_modules="all-linear"

    Questo è il modo più semplice per fare un "full LoRA finetuning" senza stage.
    """
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_cfg)
    _ensure_only_lora_trainable(lora_model)
    return lora_model


# -----------------------------
# Adapter I/O (solo chiavi LoRA)
# -----------------------------
def save_lora_adapter_only(model: torch.nn.Module, adapter_path: Path) -> None:
    """
    Salva SOLO i pesi LoRA in un file unico adapter.pt.
    """
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    state = {k: v.detach().cpu() for k, v in m.state_dict().items() if "lora_" in k}
    torch.save(state, adapter_path)
    print(f"[CKPT] Saved LoRA adapter -> {adapter_path}")


def load_lora_adapter_only(
    model: torch.nn.Module,
    adapter_path: Path,
    device: str,
    strict_shapes: bool = True,
) -> Tuple[int, int]:
    """
    Carica SOLO i pesi LoRA da adapter.pt dentro un modello che ha già LoRA attaccata.

    Ritorna: (missing, unexpected)
    missing  = quante chiavi LoRA attese dal modello non sono nel checkpoint
    unexpected = quante chiavi del checkpoint non matchano nel modello
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    ckpt = torch.load(adapter_path, map_location="cpu")

    m = model.module if hasattr(model, "module") else model
    model_state = m.state_dict()

    model_lora_keys = {k for k in model_state.keys() if "lora_" in k}
    ckpt_keys = set(ckpt.keys())

    unexpected = 0
    loaded = 0

    for k, v in ckpt.items():
        if k in model_state:
            # opzionale: controllo shape (utile per beccare mismatch di target_modules/r)
            if strict_shapes and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                if tuple(v.shape) != tuple(model_state[k].shape):
                    raise RuntimeError(
                        f"[LoRA LOAD] Shape mismatch for '{k}': ckpt={tuple(v.shape)} "
                        f"model={tuple(model_state[k].shape)}"
                    )
            model_state[k] = v.to(device)
            loaded += 1
        else:
            unexpected += 1

    m.load_state_dict(model_state, strict=False)
    missing = len(model_lora_keys - ckpt_keys)

    print(f"[LOAD] LoRA adapter loaded: loaded={loaded}, missing={missing}, unexpected={unexpected}")
    return missing, unexpected


# -----------------------------
# Main loader (MODEL + PROCESSOR + LORA)
# -----------------------------
def load_qwen3vl_full_lora(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = True,
) -> Tuple[torch.nn.Module, Any]:
    """
    Carica processor + Qwen3VLForConditionalGeneration, congela base, attacca LoRA su all-linear.

    - dtype:
        * se None: usa bf16 se disponibile su GPU (consigliato), altrimenti fp16 su GPU, fp32 su CPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype auto
    if dtype is None:
        if device.startswith("cuda"):
            # bf16 se supportato, altrimenti fp16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

    # 1) Processor
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    t1 = time.perf_counter()
    print(f"[TIMER] Processor loaded in {t1 - t0:.2f}s")

    # 2) Base model
    t2 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    t3 = time.perf_counter()
    print(f"[TIMER] Base model loaded in {t3 - t2:.2f}s (dtype={dtype}, attn={attn_implementation})")

    # training safety
    model.config.use_cache = False

    if gradient_checkpointing:
        # per Qwen3-VL di solito va bene così
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[INFO] Gradient checkpointing: ON")
    else:
        print("[INFO] Gradient checkpointing: OFF")

    # 3) Freeze base
    _freeze_all_params(model)

    # 4) Attach LoRA (all-linear)
    t4 = time.perf_counter()
    lora_model = attach_lora_all_linear(model, r=r, alpha=alpha, dropout=dropout)
    t5 = time.perf_counter()
    print(f"[TIMER] LoRA attached in {t5 - t4:.2f}s (target_modules=all-linear)")
    _print_trainable_stats(lora_model)

    # 5) To device
    lora_model.to(device)
    print(f"[INFO] Model moved to device: {device}")

    return lora_model, processor


# -----------------------------
# Minimal quick self-test
# -----------------------------
if __name__ == "__main__":
    # Piccolo smoke test di caricamento
    model, processor = load_qwen3vl_full_lora()
    print("[OK] load_qwen3vl_full_lora() completed.")