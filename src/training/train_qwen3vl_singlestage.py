#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/training/train_qwen3vl_singlestage.py

Single-stage LoRA training for Qwen3-VL on How2Sign (video->text).

- Prompting via apply_chat_template (multimodal messages: video + text)
- Robust masking using anchor "Translation:" searched INSIDE input_ids
- OVERFIT_TEST: fixed subset + val = same subset
- No checkpoints / resume / early stopping
- Clear error if torchrun requests more ranks than visible GPUs

Launch examples:
  torchrun --nproc_per_node=2 src/training/train_qwen3vl_singlestage.py
"""

import os
import sys
import time
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

# Avoid HF tokenizers fork warnings / deadlocks
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
# Single-stage: use your stage1 loader (LoRA on base) as the only loader.
# If you have a dedicated single-stage loader, swap this import accordingly.
from src.models.qwen3vl_lora import load_qwen3vl_lora


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

# Validation
MAX_VAL_BATCHES = 50

# Generation debug (all ranks run it to keep DDP in sync; only rank0 prints)
DO_GENERATION_EVAL = True
GEN_EVERY_EPOCH = True
GEN_MAX_NEW_TOKENS = 64
GEN_NUM_EXAMPLES = 5  # total printed (rank0)

# OVERFIT TEST
OVERFIT_TEST = True
OVERFIT_N_SAMPLES = 32
OVERFIT_SEED = 123

# Logging
LOG_EVERY = 2


# =====================================================================
# DDP helpers
# =====================================================================
def setup_ddp_if_needed() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return dict(rank=0, world_size=1, local_rank=0, device="cpu", is_main=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = "cuda:0"
        torch.cuda.set_device(0)
        return dict(rank=0, world_size=1, local_rank=0, device=device, is_main=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    n_gpus = torch.cuda.device_count()
    if local_rank >= n_gpus:
        # This is the "invalid device ordinal" you saw.
        raise RuntimeError(
            f"[DDP] local_rank={local_rank} but visible CUDA devices={n_gpus}. "
            "Fix: launch torchrun with --nproc_per_node equal to the number of visible GPUs "
            "(or adjust CUDA_VISIBLE_DEVICES)."
        )

    torch.cuda.set_device(local_rank)

    # Init process group
    # If your PyTorch supports device_id arg, it removes barrier() warnings.
    try:
        dist.init_process_group(backend="nccl", device_id=local_rank)
    except TypeError:
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


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM on a scalar tensor, returns reduced tensor."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# =====================================================================
# Prompt + masking (apply_chat_template + anchor-based masking)
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
    Build chat-formatted full texts (prompt + assistant target) and mask everything up to and
    including the anchor "Translation:" located INSIDE input_ids.
    """
    full_texts: List[str] = []
    for vp, tgt in zip(videos_batch, texts_batch):
        msgs = build_messages(vp, target_text=tgt)
        full_texts.append(_apply_chat_template(processor, msgs, add_generation_prompt=False))

    inputs = processor(
        videos=videos_batch,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    labels = input_ids.clone()
    ignore_index = -100

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
        if start_idx != -1:
            end_idx = start_idx + len(anchor_ids)
            labels[b, :end_idx] = ignore_index
        # else: fallback -> do not mask (better than masking all)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = ignore_index

    return inputs, labels


# =====================================================================
# Generation debug (all ranks run to keep DDP synchronized; only rank0 prints)
# =====================================================================
@torch.no_grad()
def generate_translations(
    model,
    processor,
    device: str,
    videos: List[str],
    max_new_tokens: int,
) -> List[str]:
    m = model.module if hasattr(model, "module") else model
    m.eval()

    prompts: List[str] = []
    for vp in videos:
        msgs = build_messages(vp, target_text=None)
        prompts.append(_apply_chat_template(processor, msgs, add_generation_prompt=True))

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
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # decode only continuation (avoid echoing prompt)
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
# Eval (distributed-safe)
# =====================================================================
@torch.no_grad()
def evaluate(
    model,
    processor,
    device: str,
    loader: DataLoader,
    max_batches: int,
    ddp: Dict[str, Any],
) -> Dict[str, float]:
    """
    Each rank computes partial sum(loss) and count, then all-reduce to get global average.
    """
    model.eval()

    loss_sum = 0.0
    n = 0

    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break

        videos_batch: List[str] = batch["videos"]
        texts_batch: List[str] = batch["texts"]

        inputs, labels = make_batch_inputs_and_labels(processor, videos_batch, texts_batch, device)
        outputs = model(**inputs, labels=labels)
        loss_sum += float(outputs.loss.item())
        n += 1

    # all-reduce sums
    if ddp["world_size"] > 1:
        t_sum = torch.tensor([loss_sum], device=device, dtype=torch.float32)
        t_n = torch.tensor([n], device=device, dtype=torch.float32)
        ddp_all_reduce_sum(t_sum)
        ddp_all_reduce_sum(t_n)
        loss_sum = float(t_sum.item())
        n = int(t_n.item())

    model.train()
    return {"val_loss": (loss_sum / max(n, 1)) if n > 0 else float("nan")}


@torch.no_grad()
def evaluate_generation_debug_synchronized(
    model,
    processor,
    device: str,
    loader: DataLoader,
    max_batches: int,
    ddp: Dict[str, Any],
) -> None:
    """
    To avoid DDP desync:
    - ALL ranks run the same generation loop for the same number of batches.
    - Only rank0 prints.
    """
    model.eval()
    is_main = ddp["is_main"]

    printed = 0
    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        if printed >= GEN_NUM_EXAMPLES:
            break

        videos_batch: List[str] = batch["videos"]
        texts_batch: List[str] = batch["texts"]

        preds = generate_translations(
            model=model,
            processor=processor,
            device=device,
            videos=videos_batch,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )

        # Everyone does the work, only rank0 prints
        if is_main:
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
    running_loss = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        if ddp["world_size"] > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main else train_loader

        for _, batch in enumerate(data_iter, start=1):
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

        # ---- Distributed-safe eval (all ranks participate) ----
        metrics = evaluate(
            model=model,
            processor=processor,
            device=device,
            loader=val_loader,
            max_batches=MAX_VAL_BATCHES,
            ddp=ddp,
        )
        if is_main:
            print(f"[EVAL] epoch={epoch} val_loss={metrics['val_loss']:.4f}")

        # ---- Generation debug (all ranks run; only rank0 prints) ----
        if DO_GENERATION_EVAL and GEN_EVERY_EPOCH:
            evaluate_generation_debug_synchronized(
                model=model,
                processor=processor,
                device=device,
                loader=val_loader,
                max_batches=MAX_VAL_BATCHES,
                ddp=ddp,
            )

        if MAX_STEPS is not None and global_step >= MAX_STEPS:
            break

    if is_main:
        print("\n[INFO] Training finished.")


# =====================================================================
# Main
# =====================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ddp = setup_ddp_if_needed()
    rank = ddp["rank"]
    world_size = ddp["world_size"]
    local_rank = ddp["local_rank"]
    device = ddp["device"]
    is_main = ddp["is_main"]

    if is_main:
        print("==============================================")
        print("   QWEN3-VL Single-Stage LoRA — HOW2SIGN")
        print("==============================================")
        print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"[INFO] DATASET_JSON  = {DATASET_JSON}")
        print(f"[INFO] OUTPUT_DIR   = {OUTPUT_DIR}")
        print(f"[INFO] World size   = {world_size}")
        print(f"[INFO] Rank/L-Rank   = {rank}/{local_rank}")
        print(f"[INFO] Device        = {device}")
        print(f"[INFO] OVERFIT_TEST  = {OVERFIT_TEST} (n={OVERFIT_N_SAMPLES})")
        print("==============================================\n")

    try:
        # 1) Load model + processor (single-stage = stage1)
        t0 = time.time()
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16,
            alpha=32,
            dropout=0.05,
            device=device,
            stage="stage1",
        )
        if is_main:
            print(f"[TIMER] model+processor loaded in {time.time() - t0:.2f}s")

        # 2) DDP wrap
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
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

        # 4) Samplers / Loaders
        if world_size > 1:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            shuffle_flag = False
        else:
            train_sampler = None
            val_sampler = None
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

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
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