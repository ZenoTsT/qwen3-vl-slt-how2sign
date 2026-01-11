#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/training/train_qwen3vl_singlestage.py

Single-stage LoRA training for Qwen3-VL on How2Sign (video->text).
- DDP ready (torchrun)
- OVERFIT_TEST supported (train on a fixed small subset)
- Train/Evaluate (+ optional generation debug on rank0)

Launch (example):
  torchrun --nproc_per_node=2 src/training/train_qwen3vl_singlestage.py
"""

import os
import sys
import time
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

# Avoid tokenizers fork warnings/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch import amp
from tqdm import tqdm


# ---------------------------------------------------------------------
# PYTHONPATH: add repo root
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn

# >>> IMPORTANT: this MUST match your actual loader module filename <<<
# If you truly have: src/models/qwen3vl_lora_full.py, use this:
from src.models.qwen3vl_lora_singlestage import load_qwen3vl_full_lora
# If instead you kept the old name, change the import accordingly.


# =====================================================================
# CONFIG
# =====================================================================
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

DATASET_JSON = PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset_clean.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign_singlestage"

NUM_EPOCHS = 10
BATCH_SIZE = 1
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 16
MAX_STEPS: Optional[int] = None

# Validation (rank0 only)
MAX_VAL_BATCHES = 50

# Generation debug (rank0 only)
DO_GENERATION_EVAL = True
GEN_EVERY_EPOCH = True
GEN_MAX_NEW_TOKENS = 64
GEN_NUM_EXAMPLES = 5

# OVERFIT TEST
OVERFIT_TEST = True
OVERFIT_N_SAMPLES = 32
OVERFIT_SEED = 123

# Logging
LOG_EVERY = 20


# =====================================================================
# DDP helpers
# =====================================================================
def setup_ddp() -> Dict[str, Any]:
    """
    Returns dict:
      rank, world_size, local_rank, device, is_main
    Robust against 'invalid device ordinal' when torchrun local ranks
    don't match visible GPUs.
    """
    if not torch.cuda.is_available():
        return dict(rank=0, world_size=1, local_rank=0, device="cpu", is_main=True)

    # torchrun sets these
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        return dict(rank=0, world_size=1, local_rank=0, device="cpu", is_main=True)

    # If torchrun asked for more procs than visible GPUs, map safely
    if local_rank >= n_gpu:
        mapped = local_rank % n_gpu
        if rank == 0:
            print(f"[WARN] local_rank={local_rank} but only {n_gpu} GPUs visible -> mapping to {mapped}")
        local_rank = mapped

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    return dict(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=f"cuda:{local_rank}",
        is_main=(rank == 0),
    )


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(ddp: Dict[str, Any]) -> None:
    if ddp["world_size"] > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()


# =====================================================================
# Prompt + masking
# =====================================================================
SYSTEM_PROMPT = "You are a sign language translation model."
USER_INSTRUCTION = (
    "Translate the sign language video into English.\n\n"
    "Answer with the English sentence only.\n\n"
    "Translation:"
)
ANCHOR_TEXT = "Translation:"


def build_messages(video_path: str, target_text: Optional[str] = None) -> List[Dict[str, Any]]:
    user_content = [
        {"type": "video", "video": video_path},
        {"type": "text", "text": USER_INSTRUCTION},
    ]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if target_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return msgs


def _apply_chat_template(processor, messages, add_generation_prompt: bool) -> str:
    # some processors expose apply_chat_template; fallback to tokenizer
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def make_batch_inputs_and_labels(
    processor,
    videos_batch: List[str],
    texts_batch: List[str],
    device: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Robust SFT masking for multimodal:
    - build full_text (prompt + assistant target) via chat template
    - encode with processor(videos=..., text=...)
    - locate anchor "Translation:" inside input_ids (includes video tokens inserted by processor)
    - mask labels up to (and including) the anchor
    """
    full_texts: List[str] = []
    for vp, tgt in zip(videos_batch, texts_batch):
        msg_full = build_messages(vp, target_text=tgt)
        full_texts.append(_apply_chat_template(processor, msg_full, add_generation_prompt=False))

    full_inputs = processor(
        videos=videos_batch,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    full_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full_inputs.items()}

    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    ignore_index = -100

    # Tokenize anchor once (no special tokens)
    anchor_ids = processor.tokenizer(
        ANCHOR_TEXT,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    B, T = input_ids.shape
    for b in range(B):
        seq = input_ids[b].tolist()

        start_idx = -1
        for j in range(0, T - len(anchor_ids) + 1):
            if seq[j : j + len(anchor_ids)] == anchor_ids:
                start_idx = j
                break

        if start_idx == -1:
            # If anchor not found, conservative fallback:
            # mask nothing (better than masking everything)
            continue

        end_idx = start_idx + len(anchor_ids)
        labels[b, :end_idx] = ignore_index

    # mask padding tokens
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = ignore_index

    return full_inputs, labels


# =====================================================================
# Generation debug (rank0 only)
# =====================================================================
@torch.no_grad()
def generate_translations(
    model,
    processor,
    device: str,
    videos: List[str],
) -> List[str]:
    # unwrap DDP for generation
    m = model.module if hasattr(model, "module") else model
    m.eval()

    prompts: List[str] = []
    for vp in videos:
        msg_prompt = build_messages(vp, target_text=None)
        prompts.append(_apply_chat_template(processor, msg_prompt, add_generation_prompt=True))

    inputs = processor(
        videos=videos,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    gen_ids = m.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
    )

    # Decode ONLY the generated continuation (avoid prompt echo)
    in_len = inputs["input_ids"].shape[1]
    gen_only = gen_ids[:, in_len:] if gen_ids.shape[1] > in_len else gen_ids
    decoded = processor.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    cleaned: List[str] = []
    for s in decoded:
        s2 = s.strip()
        if "\n" in s2:
            parts = [p.strip() for p in s2.split("\n") if p.strip()]
            if parts:
                s2 = parts[-1]
        cleaned.append(s2)
    return cleaned


# =====================================================================
# Eval (rank0 only)
# =====================================================================
@torch.no_grad()
def evaluate_loss(
    model,
    processor,
    device: str,
    loader: DataLoader,
    max_batches: int,
    ddp: Dict[str, Any],
) -> Dict[str, float]:
    if not ddp["is_main"]:
        return {}

    model.eval()
    losses: List[float] = []

    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break

        videos_batch: List[str] = batch["videos"]
        texts_batch: List[str] = batch["texts"]

        inputs, labels = make_batch_inputs_and_labels(processor, videos_batch, texts_batch, device)
        outputs = model(**inputs, labels=labels)
        losses.append(float(outputs.loss.item()))

    model.train()
    return {"loss": float(sum(losses) / max(len(losses), 1)) if losses else float("nan")}


@torch.no_grad()
def evaluate_generation_debug(
    model,
    processor,
    device: str,
    loader: DataLoader,
    max_batches: int,
    ddp: Dict[str, Any],
) -> None:
    if not ddp["is_main"]:
        return

    model.eval()
    printed = 0

    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches or printed >= GEN_NUM_EXAMPLES:
            break

        videos_batch: List[str] = batch["videos"]
        texts_batch: List[str] = batch["texts"]

        preds = generate_translations(model, processor, device, videos_batch)
        for ref, pred in zip(texts_batch, preds):
            print("----- GEN DEBUG -----")
            print("REF :", ref)
            print("PRED:", pred)
            print("---------------------\n")
            printed += 1
            if printed >= GEN_NUM_EXAMPLES:
                break

    model.train()


# =====================================================================
# Dataset helpers (overfit subset)
# =====================================================================
def make_overfit_subset(ds, n: int, seed: int) -> Subset:
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    idx.sort()
    return Subset(ds, idx)


# =====================================================================
# Train loop
# =====================================================================
def train_loop(
    model,
    processor,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ddp: Dict[str, Any],
):
    is_main = ddp["is_main"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = amp.GradScaler("cuda") if device.startswith("cuda") else None
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if ddp["world_size"] > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main else train_loader
        running_loss = 0.0

        for step_in_epoch, batch in enumerate(data_iter, start=1):
            global_step += 1

            videos_batch: List[str] = batch["videos"]
            texts_batch: List[str] = batch["texts"]

            inputs, labels = make_batch_inputs_and_labels(processor, videos_batch, texts_batch, device)

            autocast_ctx = torch.amp.autocast("cuda") if device.startswith("cuda") else nullcontext()
            with autocast_ctx:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * GRAD_ACCUM_STEPS

            if global_step % GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            if is_main and (global_step % LOG_EVERY == 0):
                avg = running_loss / max(1, LOG_EVERY)
                running_loss = 0.0
                if isinstance(data_iter, tqdm):
                    data_iter.set_postfix({"loss": f"{avg:.4f}"})
                print(f"[STEP {global_step}] loss={avg:.4f}")

            if MAX_STEPS is not None and global_step >= MAX_STEPS:
                if is_main:
                    print(f"[INFO] Reached MAX_STEPS={MAX_STEPS} -> stopping.")
                break

        # -------------------------------
        # IMPORTANT: DDP-safe evaluation
        # -------------------------------
        # Barrier BEFORE eval: all ranks must reach this point
        ddp_barrier(ddp)

        # Eval/generation only on rank0
        if is_main:
            ev = evaluate_loss(model, processor, device, val_loader, MAX_VAL_BATCHES, ddp)
            print(f"[EVAL] epoch={epoch} val_loss={ev.get('loss', float('nan')):.4f}")

            if DO_GENERATION_EVAL and GEN_EVERY_EPOCH:
                evaluate_generation_debug(model, processor, device, val_loader, max_batches=MAX_VAL_BATCHES, ddp=ddp)

        # Barrier AFTER eval: all ranks wait for rank0 to finish extra work
        ddp_barrier(ddp)

        if MAX_STEPS is not None and global_step >= MAX_STEPS:
            break

    if is_main:
        print("\n[INFO] Training finished.")


# =====================================================================
# Main
# =====================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ddp = setup_ddp()
    is_main = ddp["is_main"]
    device = ddp["device"]

    if is_main:
        print("==============================================")
        print("   QWEN3-VL Single-Stage LoRA — HOW2SIGN")
        print("==============================================")
        print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"[INFO] DATASET_JSON  = {DATASET_JSON}")
        print(f"[INFO] OUTPUT_DIR   = {OUTPUT_DIR}")
        print(f"[INFO] device       = {device}")
        print(f"[INFO] world_size   = {ddp['world_size']}")
        print(f"[INFO] OVERFIT_TEST = {OVERFIT_TEST} (n={OVERFIT_N_SAMPLES})")
        print("==============================================\n")

    try:
        # 1) Load model + processor (per-rank device)
        t0 = time.time()
        model, processor = load_qwen3vl_full_lora(
            model_name=MODEL_NAME,
            r=16,
            alpha=32,
            dropout=0.05,
            device=device,
        )
        if is_main:
            print(f"[TIMER] model+processor loaded in {time.time() - t0:.2f}s")

        # 2) DDP wrap
        if ddp["world_size"] > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ddp["local_rank"]],
                output_device=ddp["local_rank"],
                find_unused_parameters=False,
            )

        # 3) Dataset
        train_ds_full = How2SignDataset(
            json_path=str(DATASET_JSON),
            split="train",
            root_dir=str(PROJECT_ROOT),
            return_type="video",
        )

        if OVERFIT_TEST:
            train_ds = make_overfit_subset(train_ds_full, OVERFIT_N_SAMPLES, OVERFIT_SEED)
            val_ds = train_ds
        else:
            train_ds = train_ds_full
            val_ds = How2SignDataset(
                json_path=str(DATASET_JSON),
                split="val",
                root_dir=str(PROJECT_ROOT),
                return_type="video",
            )

        if is_main:
            print(f"[DATA] train={len(train_ds)} | val={len(val_ds)}")

        # 4) Loaders
        if ddp["world_size"] > 1:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=ddp["world_size"],
                rank=ddp["rank"],
                shuffle=True,
            )
            shuffle_flag = False
        else:
            train_sampler = None
            shuffle_flag = True

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle_flag,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            collate_fn=how2sign_collate_fn,
            pin_memory=device.startswith("cuda"),
        )

        # rank0 eval only -> no sampler needed; still safe due to barriers
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            collate_fn=how2sign_collate_fn,
            pin_memory=device.startswith("cuda"),
        )

        # 5) Train
        train_loop(
            model=model,
            processor=processor,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            ddp=ddp,
        )

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()