#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval script for Qwen3-VL LoRA checkpoints (How2Sign).
- Computes BLEU-1/2/3/4 and ROUGE (rouge1/rouge2/rougeL).
- Supports:
    * stage1: load Stage1 LoRA adapter
    * stage2: load Stage1 LoRA (merged/frozen) + Stage2 LoRA adapter (simultaneously, stacked)
- Saves per-sample {video_path, ref, pred} to a JSONL file.

Example (stage2 on val):
python scripts/eval_checkpoints.py \
  --stage stage2 \
  --split val \
  --max_samples 200 \
  --stage1_dir outputs/qwen3vl_lora_how2sign/checkpoints/stage1/epoch_best \
  --stage2_dir outputs/qwen3vl_lora_how2sign/checkpoints/stage2/intra_latest \
  --out_jsonl outputs/qwen3vl_lora_how2sign/logs/eval_stage2_val.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# PYTHONPATH: add repo root so `from src...` works
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = THIS_DIR.parent                      # .../ (repo root)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn  # noqa: E402
from src.models.qwen3vl_lora import load_qwen3vl_lora                          # noqa: E402


# ---------------------------------------------------------------------
# Prompt: MUST match the one used during training
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
# Metrics deps
# ---------------------------------------------------------------------
def _ensure_metrics_libs():
    try:
        import sacrebleu  # noqa
    except Exception:
        raise RuntimeError(
            "Missing dependency: sacrebleu\n"
            "Install with:\n"
            "  pip install --user sacrebleu\n"
        )
    try:
        from rouge_score import rouge_scorer  # noqa
    except Exception:
        raise RuntimeError(
            "Missing dependency: rouge-score\n"
            "Install with:\n"
            "  pip install --user rouge-score\n"
        )


# ---------------------------------------------------------------------
# BLEU / ROUGE implementations (robust to sacrebleu versions)
# ---------------------------------------------------------------------
def compute_bleu_1_4(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu

    # sacrebleu expects list of hypotheses and list-of-lists of references
    ref_list = [refs]

    def bleu_with_order(order: int) -> float:
        # Different sacrebleu versions have slightly different BLEU() signatures.
        # We avoid passing use_effective_order (it broke on your cluster),
        # and keep it simple + stable.
        try:
            metric = sacrebleu.metrics.BLEU(
                ngram_order=order,
                smooth_method="exp",
                smooth_value=None,
                force=False,
                tokenize="13a",
                lowercase=False,
            )
        except TypeError:
            # Older versions might not support some kwargs
            metric = sacrebleu.metrics.BLEU(
                ngram_order=order,
                smooth_method="exp",
                tokenize="13a",
            )

        score = metric.corpus_score(preds, ref_list).score
        return float(score)

    return {
        "BLEU1": bleu_with_order(1),
        "BLEU2": bleu_with_order(2),
        "BLEU3": bleu_with_order(3),
        "BLEU4": bleu_with_order(4),
    }


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_f, r2_f, rl_f = 0.0, 0.0, 0.0
    n = max(1, len(preds))

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # score(reference, prediction)
        r1_f += s["rouge1"].fmeasure
        r2_f += s["rouge2"].fmeasure
        rl_f += s["rougeL"].fmeasure

    return {
        "ROUGE1_F": float(r1_f / n),
        "ROUGE2_F": float(r2_f / n),
        "ROUGEL_F": float(rl_f / n),
    }


# ---------------------------------------------------------------------
# Loading model with correct "stacking" of stage1+stage2 adapters
# ---------------------------------------------------------------------
def load_model_for_eval(
    model_name: str,
    stage: str,
    device: str,
    stage1_dir: str | None,
    stage2_dir: str | None,
):
    """
    stage1:
      - attaches stage1 LoRA, then loads stage1 adapter weights
    stage2:
      - base model
      - loads stage1 LoRA weights and merges/freeze into base (inside load_qwen3vl_lora stage2)
      - attaches stage2 LoRA
      - loads stage2 adapter weights
    """
    stage = stage.lower().strip()
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Invalid stage: {stage}")

    if stage == "stage1":
        if stage1_dir is None:
            raise ValueError("stage1 requires --stage1_dir")
        model, processor = load_qwen3vl_lora(
            model_name=model_name,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
        # load adapter.pt into attached LoRA weights
        adapter_path = Path(stage1_dir) / "adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing stage1 adapter.pt at: {adapter_path}")
        state = torch.load(adapter_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[CKPT] Stage1 adapter loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        return model, processor

    # stage2
    if stage1_dir is None or stage2_dir is None:
        raise ValueError("stage2 requires BOTH --stage1_dir and --stage2_dir")

    model, processor = load_qwen3vl_lora(
        model_name=model_name,
        r=16, alpha=32, dropout=0.05,
        device=device,
        stage="stage2",
        stage1_adapter_dir=str(stage1_dir),  # <-- this is where stage1 is merged/frozen
    )

    adapter_path = Path(stage2_dir) / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing stage2 adapter.pt at: {adapter_path}")

    state = torch.load(adapter_path, map_location="cpu")
    # The stage2 model is a PeftModel; loading LoRA keys should work non-strict
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Stage2 adapter loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    return model, processor


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_predictions(
    model,
    processor,
    dataloader,
    device: str,
    max_samples: int,
    max_new_tokens: int,
) -> Tuple[List[str], List[str], List[Dict]]:
    model.eval()

    preds_all: List[str] = []
    refs_all: List[str] = []
    rows: List[Dict] = []

    prompt = build_instruction_prompt()

    seen = 0
    t0 = time.time()

    for batch in tqdm(dataloader, desc="Evaluating", total=max_samples if max_samples else None):
        videos = batch["videos"]   # List[str]
        refs = batch["texts"]      # List[str]

        # Stop if reached max_samples
        if max_samples is not None and max_samples > 0:
            if seen >= max_samples:
                break
            # If batch would exceed, truncate
            if seen + len(videos) > max_samples:
                cut = max_samples - seen
                videos = videos[:cut]
                refs = refs[:cut]

        prompts = [prompt for _ in videos]

        inputs = processor(
            videos=videos,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic
        )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for vp, ref, pred in zip(videos, refs, decoded):
            # keep only what comes after "Translation:" if present
            pred_clean = pred.split("Translation:")[-1].strip()
            preds_all.append(pred_clean)
            refs_all.append(ref)
            rows.append({"video_path": vp, "ref": ref, "pred": pred_clean})

        seen += len(videos)

    t1 = time.time()
    print(f"[INFO] Generation done in {t1 - t0:.1f}s for {seen} samples.")
    return preds_all, refs_all, rows


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def save_jsonl(rows: List[Dict], out_path: str):
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved predictions to: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--json_path", type=str, default="/work/tesi_ztesta/How2Sign_resized/how2sign_dataset_clean.json")
    p.add_argument("--root_dir", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")

    # IMPORTANT: must be provided (you hit this error)
    p.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2"])

    # checkpoint dirs (directories that contain adapter.pt)
    p.add_argument("--stage1_dir", type=str, default=None)
    p.add_argument("--stage2_dir", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--out_jsonl", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    _ensure_metrics_libs()

    # Output jsonl default name
    if args.out_jsonl is None:
        out_dir = Path(PROJECT_ROOT) / "outputs" / "qwen3vl_lora_how2sign" / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out_jsonl = str(out_dir / f"eval_{args.stage}_{args.split}.jsonl")

    print("[INFO] Running eval:")
    print(f"  stage      = {args.stage}")
    print(f"  split      = {args.split}")
    print(f"  max_samples= {args.max_samples}")
    print(f"  stage1_dir = {args.stage1_dir}")
    print(f"  stage2_dir = {args.stage2_dir}")
    print(f"  out_jsonl  = {args.out_jsonl}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Load model + processor (with correct stage stacking)
    model, processor = load_model_for_eval(
        model_name=args.model_name,
        stage=args.stage,
        device=device,
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
    )

    # Dataset / loader
    ds = How2SignDataset(
        json_path=args.json_path,
        split=args.split,
        root_dir=args.root_dir,
        return_type="video",
    )
    print(f"[How2SignDataset] split={args.split} | num_samples={len(ds)}")
    print(f"[How2SignDataset] json_path={args.json_path}")
    print(f"[How2SignDataset] root_dir={args.root_dir}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=how2sign_collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Generate
    preds, refs, rows = generate_predictions(
        model=model,
        processor=processor,
        dataloader=dl,
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Save predictions
    save_jsonl(rows, args.out_jsonl)

    # Metrics
    bleu = compute_bleu_1_4(preds, refs)
    rouge = compute_rouge(preds, refs)

    metrics = {**bleu, **rouge}
    print("\n================= METRICS =================")
    for k, v in metrics.items():
        # BLEU in [0,100], ROUGE_F in [0,1]
        if k.startswith("ROUGE"):
            print(f"{k:10s}: {v:.4f}")
        else:
            print(f"{k:10s}: {v:.2f}")
    print("===========================================\n")

    # Also save metrics next to jsonl
    metrics_path = str(Path(args.out_jsonl).with_suffix(".metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()