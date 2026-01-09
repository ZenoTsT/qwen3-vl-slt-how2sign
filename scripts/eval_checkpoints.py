#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script with robust checkpoint selection logic:

CHECKPOINT POLICY (as requested)
--------------------------------
Stage1 checkpoint resolution priority:
  1) epoch_best
  2) epoch_latest
  3) intra_latest
  else: none

Stage2 checkpoint resolution priority:
  1) epoch_best
  2) epoch_latest
  3) intra_latest
  else: none

Decision:
- If Stage1 is ONLY intra_latest (no epoch_best/epoch_latest):
    -> evaluate using Stage1 intra_latest ONLY
    -> DO NOT load Stage2 even if present
- If Stage1 is epoch_best/epoch_latest:
    - If Stage2 exists (any kind):
        -> load Stage2 pipeline:
           (load base) + (attach Stage1 LoRA + load Stage1 adapter) + (MERGE) + (attach Stage2 LoRA) + (load Stage2 adapter)
    - Else:
        -> evaluate using Stage1 only (epoch checkpoint)

Also prints extra debug to understand why generation might look "empty".
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# PYTHONPATH: add repo root
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent                 # .../scripts
PROJECT_ROOT = THIS_DIR.parent                             # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn
from src.models.qwen3vl_lora import load_qwen3vl_lora


# ---------------------------------------------------------------------
# HARD-CODED CONFIG
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
SPLIT = "test"
MAX_SAMPLES = 20

# Generation
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

DEBUG_FIRST_N = 5

DATASET_JSON = PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset_clean.json"
ROOT_DIR = PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign"
LOGS_DIR = OUTPUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = LOGS_DIR / f"eval_auto_{SPLIT}_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"

# Where checkpoints live
CKPT_ROOT = OUTPUT_DIR / "checkpoints"
STAGE1_ROOT = CKPT_ROOT / "stage1"
STAGE2_ROOT = CKPT_ROOT / "stage2"

# Loader
NUM_WORKERS = 2
BATCH_SIZE = 1


# ---------------------------------------------------------------------
# Prompt (must match training instruction)
# ---------------------------------------------------------------------
def build_instruction_prompt() -> str:
    return (
        "You are a sign language translation model. "
        "Given the following sign language video <|video_pad|>, "
        "translate it into English.\n\n"
        "Answer with the English sentence only.\n\n"
        "Translation:"
    )


# ---------------------------------------------------------------------
# Checkpoint resolving
# ---------------------------------------------------------------------
def _ckpt_dir(stage_root: Path, kind: str) -> Path:
    return stage_root / kind

def resolve_checkpoint(stage_root: Path) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
    """
    Returns: (kind, adapter_path, state_path) using priority:
      epoch_best > epoch_latest > intra_latest
    """
    priority = ["epoch_best", "epoch_latest", "intra_latest"]
    for kind in priority:
        d = _ckpt_dir(stage_root, kind)
        a = d / "adapter.pt"
        s = d / "state.pt"
        if a.exists() and s.exists():
            return kind, a, s
    return None, None, None

def resolve_checkpoint_dir(stage_root: Path, kind: str) -> Path:
    return _ckpt_dir(stage_root, kind)

def load_adapter_into_model_only_lora(model, adapter_path: Path, device: str) -> Tuple[int, int]:
    """
    Loads only LoRA weights from adapter.pt into an already PEFT-instrumented model.
    Returns (missing_lora_keys, unexpected_keys).
    """
    adapter_state = torch.load(adapter_path, map_location="cpu")

    model_to_load = model.module if hasattr(model, "module") else model
    model_state = model_to_load.state_dict()

    model_lora_keys = {k for k in model_state.keys() if "lora_" in k}
    adapter_keys = set(adapter_state.keys())

    unexpected = 0
    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k] = v.to(device)
        else:
            unexpected += 1

    model_to_load.load_state_dict(model_state, strict=False)

    missing = len(model_lora_keys - adapter_keys)
    return missing, unexpected


# ---------------------------------------------------------------------
# Model loading policy
# ---------------------------------------------------------------------
def build_model_with_correct_adapters(device: str):
    """
    Implements the requested policy:
    - stage1 intra only -> stage1 only
    - stage1 epoch + stage2 exists -> merge stage1 then stage2 adapter
    - stage1 epoch only -> stage1 only
    - stage1 missing -> if stage2 exists, stage2 standalone (fallback)
    """
    s1_kind, s1_adapter, s1_state = resolve_checkpoint(STAGE1_ROOT)
    s2_kind, s2_adapter, s2_state = resolve_checkpoint(STAGE2_ROOT)

    # Additionally: detect "stage1 is only intra" case explicitly
    s1_has_epoch_best = (resolve_checkpoint_dir(STAGE1_ROOT, "epoch_best") / "adapter.pt").exists()
    s1_has_epoch_latest = (resolve_checkpoint_dir(STAGE1_ROOT, "epoch_latest") / "adapter.pt").exists()
    s1_has_epoch = s1_has_epoch_best or s1_has_epoch_latest
    s1_has_intra = (resolve_checkpoint_dir(STAGE1_ROOT, "intra_latest") / "adapter.pt").exists()

    # Compute "stage1 only intra" (no epoch checkpoint but intra exists)
    s1_only_intra = (not s1_has_epoch) and s1_has_intra

    print("============================================================")
    print("[CKPT RESOLVE]")
    print(f"  Stage1 resolved kind: {s1_kind} | adapter: {s1_adapter}")
    print(f"  Stage2 resolved kind: {s2_kind} | adapter: {s2_adapter}")
    print(f"  Stage1 has_epoch? {s1_has_epoch} | has_intra? {s1_has_intra} | only_intra? {s1_only_intra}")
    print("============================================================\n")

    # -----------------------
    # CASE A: Stage1 only intra -> evaluate Stage1 only (ignore Stage2)
    # -----------------------
    if s1_only_intra:
        print("[POLICY] Stage1 has ONLY intra_latest -> evaluate with Stage1 intra_latest ONLY. Stage2 will be ignored.\n")
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
        # load stage1 intra adapter
        s1_intra_adapter = resolve_checkpoint_dir(STAGE1_ROOT, "intra_latest") / "adapter.pt"
        missing, unexpected = load_adapter_into_model_only_lora(model, s1_intra_adapter, device=device)
        print(f"[LOAD] Stage1 intra_latest adapter loaded. missing={missing}, unexpected={unexpected}\n")
        return model, processor, {"mode": "stage1_only_intra", "stage1_kind": "intra_latest", "stage2_kind": None}

    # -----------------------
    # CASE B: Stage1 epoch exists
    #   - if Stage2 exists -> merge stage1 then stage2 adapter
    #   - else -> stage1 only
    # -----------------------
    if s1_has_epoch:
        # pick which epoch dir to use for stage1 merging/loading
        s1_epoch_dir = None
        if s1_has_epoch_best:
            s1_epoch_dir = resolve_checkpoint_dir(STAGE1_ROOT, "epoch_best")
            s1_epoch_kind = "epoch_best"
        else:
            s1_epoch_dir = resolve_checkpoint_dir(STAGE1_ROOT, "epoch_latest")
            s1_epoch_kind = "epoch_latest"

        if s2_kind is not None and s2_adapter is not None:
            print("[POLICY] Stage1 has epoch checkpoint AND Stage2 exists -> merge Stage1 into base, then attach Stage2 and load Stage2 adapter.\n")
            model, processor = load_qwen3vl_lora(
                model_name=MODEL_NAME,
                r=16, alpha=32, dropout=0.05,
                device=device,
                stage="stage2",
                stage1_adapter_dir=str(s1_epoch_dir),
            )
            # Now load stage2 adapter weights into the stage2 PEFT model
            missing, unexpected = load_adapter_into_model_only_lora(model, s2_adapter, device=device)
            print(f"[LOAD] Stage2 adapter loaded ({s2_kind}). missing={missing}, unexpected={unexpected}\n")
            return model, processor, {"mode": "stage1_merge_stage2", "stage1_kind": s1_epoch_kind, "stage2_kind": s2_kind}

        print("[POLICY] Stage1 has epoch checkpoint but Stage2 not found -> evaluate Stage1 only.\n")
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
        # load stage1 epoch adapter into stage1 PEFT model
        s1_epoch_adapter = s1_epoch_dir / "adapter.pt"
        missing, unexpected = load_adapter_into_model_only_lora(model, s1_epoch_adapter, device=device)
        print(f"[LOAD] Stage1 epoch adapter loaded ({s1_epoch_kind}). missing={missing}, unexpected={unexpected}\n")
        return model, processor, {"mode": "stage1_only_epoch", "stage1_kind": s1_epoch_kind, "stage2_kind": None}

    # -----------------------
    # CASE C: No Stage1 checkpoint at all
    #   - if Stage2 exists -> stage2 standalone (fallback)
    #   - else -> base model only
    # -----------------------
    if s2_kind is not None and s2_adapter is not None:
        print("[POLICY] No Stage1 checkpoint found, but Stage2 exists -> evaluating Stage2 standalone (fallback).\n")
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage2",
            stage1_adapter_dir=None,
        )
        missing, unexpected = load_adapter_into_model_only_lora(model, s2_adapter, device=device)
        print(f"[LOAD] Stage2 adapter loaded ({s2_kind}). missing={missing}, unexpected={unexpected}\n")
        return model, processor, {"mode": "stage2_standalone", "stage1_kind": None, "stage2_kind": s2_kind}

    print("[POLICY] No Stage1/Stage2 checkpoints found -> evaluating base model only.\n")
    model, processor = load_qwen3vl_lora(
        model_name=MODEL_NAME,
        r=16, alpha=32, dropout=0.05,
        device=device,
        stage="stage1",  # stage doesn't matter too much; but this attaches LoRA modules. If you want truly base-only, you'd load the HF model directly.
    )
    return model, processor, {"mode": "no_ckpt_found", "stage1_kind": None, "stage2_kind": None}


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
@torch.no_grad()
def run_generation(model, processor, device: str, loader: DataLoader) -> Dict[str, Any]:
    model.eval()

    eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)

    preds: List[str] = []
    refs: List[str] = []
    raws: List[str] = []

    n_done = 0
    t0 = time.time()

    use_autocast = device.startswith("cuda")
    autocast_ctx = torch.amp.autocast("cuda") if use_autocast else nullcontext()

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        pbar = tqdm(loader, total=min(MAX_SAMPLES, len(loader.dataset)), desc="Evaluating", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            if n_done >= MAX_SAMPLES:
                break

            videos_batch: List[str] = batch["videos"]
            texts_batch: List[str] = batch["texts"]

            prompt = build_instruction_prompt()
            prompts = [prompt] * len(videos_batch)

            inputs = processor(
                videos=videos_batch,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Debug processor output once
            if batch_idx == 0:
                print("\n================ DEBUG: PROCESSOR OUTPUT (BATCH 0) ================")
                print("processor keys:", list(inputs.keys()))
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                print("===================================================================\n")

            # Move tensors
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
            )
            if DO_SAMPLE:
                gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P))
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = eos_id
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

            with autocast_ctx:
                generated_ids = model.generate(**inputs, **gen_kwargs)

            # RAW decode (keep specials for debugging)
            decoded_raw = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            decoded_clean = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            cleaned: List[str] = []
            for s in decoded_clean:
                s2 = s.strip()
                if "Translation:" in s2:
                    s2 = s2.split("Translation:", 1)[-1].strip()
                cleaned.append(s2)

            # Extra debug: did we actually generate new tokens?
            new_tok_lens = []
            if input_len is not None:
                for i in range(generated_ids.shape[0]):
                    new_tok_lens.append(int(generated_ids[i].shape[0] - input_len))

            for vp, ref, pred, raw in zip(videos_batch, texts_batch, cleaned, decoded_raw):
                preds.append(pred)
                refs.append(ref)
                raws.append(raw)

                payload = {
                    "video_path": vp,
                    "ref": ref,
                    "pred": pred,
                    "raw": raw,
                    "input_len": input_len,
                    "total_len": int(generated_ids.shape[1]) if generated_ids.ndim == 2 else None,
                    "new_tokens": (new_tok_lens[0] if len(new_tok_lens) else None),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                n_done += 1

                if n_done <= DEBUG_FIRST_N:
                    print("\n---------------- SAMPLE DEBUG ----------------")
                    print("VIDEO:", vp)
                    print("REF :", ref)
                    print("PRED(cleaned):", repr(pred))
                    if input_len is not None:
                        print(f"LEN: input={input_len} total={generated_ids.shape[1]} new={new_tok_lens[0]}")
                    # show tail of raw to see what happens after Translation:
                    tail = raw[-400:] if len(raw) > 400 else raw
                    print("RAW tail:", tail.replace("\n", "\\n"))
                    print("---------------------------------------------\n")

            pbar.set_postfix({"done": n_done})

    dt = time.time() - t0
    return {"n_samples": n_done, "seconds": dt, "preds": preds, "refs": refs, "raws": raws}


# small helper for autocast on CPU
from contextlib import nullcontext


def main():
    print("[INFO] Eval script with auto checkpoint logic")
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATASET_JSON  = {DATASET_JSON}")
    print(f"[INFO] OUT_JSONL     = {OUT_JSONL}")
    print(f"[INFO] CKPT_ROOT     = {CKPT_ROOT}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}\n")

    # Load model+processor according to policy
    t0 = time.time()
    model, processor, policy = build_model_with_correct_adapters(device=device)
    print(f"[TIMER] model+processor ready in {time.time() - t0:.2f}s")
    print(f"[INFO] policy: {policy}\n")

    # Dataset
    ds = How2SignDataset(
        json_path=str(DATASET_JSON),
        split=SPLIT,
        root_dir=str(ROOT_DIR),
        return_type="video",
    )
    print(f"[How2SignDataset] split={SPLIT} | num_samples={len(ds)}")
    print(f"[How2SignDataset] json_path={DATASET_JSON}")
    print(f"[How2SignDataset] root_dir={ROOT_DIR}\n")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=how2sign_collate_fn,
        pin_memory=(device == "cuda"),
    )

    gen = run_generation(model, processor, device=device, loader=loader)
    print(f"\n[INFO] Generation done in {gen['seconds']:.1f}s for {gen['n_samples']} samples.")
    print(f"[INFO] Saved predictions to: {OUT_JSONL}\n")

    # quick sanity summary
    num_empty = sum(1 for p in gen["preds"] if (p.strip() == ""))
    print("=============== QUICK SANITY ===============")
    print(f"mode            : {policy['mode']}")
    print(f"stage1_kind      : {policy.get('stage1_kind')}")
    print(f"stage2_kind      : {policy.get('stage2_kind')}")
    print(f"empty_clean_preds: {num_empty}/{len(gen['preds'])}")
    print("===========================================\n")


if __name__ == "__main__":
    main()