#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval script for Qwen3-VL + LoRA checkpoints (How2Sign).
- Loads Stage1 LoRA (optional) + Stage2 LoRA (optional) using your load_qwen3vl_lora()
- Runs generation on a split (val/test/train) and saves JSONL with {video_path, ref, pred}
- Computes BLEU1..4 + ROUGE-L **without sacrebleu** (so it's version-proof)
- Adds a LOT of debug prints to verify the video actually reaches generate()

USAGE EXAMPLE (similar to your logs):
python scripts/eval_checkpoints.py \
  --stage stage2 \
  --split test \
  --max_samples 20 \
  --stage1_dir outputs/qwen3vl_lora_how2sign/checkpoints/stage1/epoch_best \
  --stage2_dir outputs/qwen3vl_lora_how2sign/checkpoints/stage2/intra_latest \
  --out_jsonl outputs/qwen3vl_lora_how2sign/logs/eval_stage2_test_XXXX.jsonl

NOTES:
- This script assumes your dataset returns "videos" as list[str] paths, as in your training.
- It assumes processor supports processor(videos=..., text=..., return_tensors="pt", padding=True, truncation=True)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# PYTHONPATH: add repo root
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent                 # .../scripts
PROJECT_ROOT = THIS_DIR.parent                             # repo root
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn
from src.models.qwen3vl_lora import load_qwen3vl_lora


# ---------------------------------------------------------------------
# Prompt (must match what you used in training for the instruction)
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
# Metrics (NO sacrebleu dependency) -> BLEU1..4 + ROUGE-L
# ---------------------------------------------------------------------
def _tokenize(s: str) -> List[str]:
    # Simple + stable tokenizer (space split). Good enough for debugging/benchmark baseline.
    # If you want 13a tokenization later, we can implement it too.
    return s.strip().split()


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _modified_precision(candidate: List[str], references: List[List[str]], n: int) -> Tuple[int, int]:
    """
    Returns (clipped_match_count, total_candidate_ngrams)
    """
    cand_counts = _ngram_counts(candidate, n)
    if not cand_counts:
        return (0, 0)

    # max reference counts per ngram
    max_ref_counts: Dict[Tuple[str, ...], int] = {}
    for ref in references:
        ref_counts = _ngram_counts(ref, n)
        for ng, c in ref_counts.items():
            prev = max_ref_counts.get(ng, 0)
            if c > prev:
                max_ref_counts[ng] = c

    clipped = 0
    total = 0
    for ng, c in cand_counts.items():
        clipped += min(c, max_ref_counts.get(ng, 0))
        total += c
    return clipped, total


def _closest_ref_length(cand_len: int, ref_lens: List[int]) -> int:
    # Standard BLEU tie-breaking: choose closest, if tie choose shortest
    best = ref_lens[0]
    best_dist = abs(best - cand_len)
    for rl in ref_lens[1:]:
        dist = abs(rl - cand_len)
        if dist < best_dist or (dist == best_dist and rl < best):
            best = rl
            best_dist = dist
    return best


def corpus_bleu_n(
    preds: List[str],
    refs: List[str],
    n: int,
    smoothing_eps: float = 1e-9,
) -> float:
    """
    Corpus BLEU with max order n (BLEU-n).
    - Geometric mean over p1..pn
    - Brevity penalty
    - Add-epsilon smoothing on precisions to avoid log(0)
    """
    assert len(preds) == len(refs)
    if len(preds) == 0:
        return float("nan")

    # Accumulate modified precisions across corpus
    p_num = [0] * n
    p_den = [0] * n
    cand_len_sum = 0
    ref_len_sum = 0

    for pred, ref in zip(preds, refs):
        cand_tok = _tokenize(pred)
        ref_tok = _tokenize(ref)

        cand_len_sum += len(cand_tok)
        ref_len_sum += _closest_ref_length(len(cand_tok), [len(ref_tok)])

        for k in range(1, n + 1):
            cm, tot = _modified_precision(cand_tok, [ref_tok], k)
            p_num[k - 1] += cm
            p_den[k - 1] += tot

    # If candidate length is 0, BLEU is 0
    if cand_len_sum == 0:
        return 0.0

    # Precisions
    import math
    log_p_sum = 0.0
    for k in range(n):
        if p_den[k] == 0:
            # no k-grams at all -> precision is 0 (smoothed)
            p = smoothing_eps
        else:
            p = p_num[k] / p_den[k]
            if p == 0.0:
                p = smoothing_eps
        log_p_sum += math.log(p)

    geo_mean = math.exp(log_p_sum / n)

    # Brevity penalty
    if cand_len_sum > ref_len_sum:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len_sum / max(cand_len_sum, 1)))

    return float(bp * geo_mean)


def _lcs_length(a: List[str], b: List[str]) -> int:
    # Dynamic programming LCS length (O(len(a)*len(b))) — OK for short sentences
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]


def corpus_rouge_l(preds: List[str], refs: List[str]) -> float:
    """
    Simple corpus ROUGE-L F1 averaged over samples.
    """
    assert len(preds) == len(refs)
    if len(preds) == 0:
        return float("nan")

    f1s = []
    for pred, ref in zip(preds, refs):
        p_tok = _tokenize(pred)
        r_tok = _tokenize(ref)
        if len(p_tok) == 0 or len(r_tok) == 0:
            f1s.append(0.0)
            continue
        lcs = _lcs_length(p_tok, r_tok)
        prec = lcs / len(p_tok)
        rec = lcs / len(r_tok)
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = (2 * prec * rec) / (prec + rec)
        f1s.append(f1)

    return float(sum(f1s) / len(f1s))


# ---------------------------------------------------------------------
# Helpers: loading LoRA adapters (only keys containing "lora_")
# ---------------------------------------------------------------------
def load_adapter_into_model_only_lora(model, adapter_path: Path, device: str) -> Tuple[int, int]:
    """
    Loads adapter.pt containing ONLY lora_ weights into model.state_dict().

    Returns: (missing_count_like, unexpected_count_like) approximations:
      - missing: how many lora_ keys in model are not provided
      - unexpected: how many lora_ keys in adapter are not found in model
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
# Core eval
# ---------------------------------------------------------------------
@torch.no_grad()
def run_generation(
    model,
    processor,
    device: str,
    loader: DataLoader,
    out_jsonl: Path,
    max_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    debug_first_n: int,
) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    preds: List[str] = []
    refs: List[str] = []

    model.eval()

    # Try to find eos token id
    eos_id = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
        eos_id = processor.tokenizer.eos_token_id

    n_done = 0
    t0 = time.time()

    with out_jsonl.open("w", encoding="utf-8") as f:
        pbar = tqdm(loader, total=min(max_samples, len(loader.dataset)), desc="Evaluating", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            if n_done >= max_samples:
                break

            videos_batch: List[str] = batch["videos"]
            texts_batch: List[str] = batch["texts"]

            # BATCH_SIZE in eval is normally 1; still we handle list
            prompt = build_instruction_prompt()
            prompts = [prompt] * len(videos_batch)

            # --- IMPORTANT: processor must create pixel_values_videos (or equivalent) ---
            inputs = processor(
                videos=videos_batch,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # -------------------- DEBUG (processor outputs) --------------------
            if batch_idx == 0:
                print("\n================ DEBUG: PROCESSOR OUTPUT (BATCH 0) ================")
                print("processor keys:", list(inputs.keys()))
                for k, v in inputs.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype',None)}")
                if "pixel_values_videos" not in inputs:
                    print("[WARN] 'pixel_values_videos' NOT found in processor output. "
                          "This often means you're generating text-only (video ignored).")
                print("===================================================================\n")

            # Move tensors to device
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            # -------------------- DEBUG: verify video tensor reaches generate --------------------
            if batch_idx == 0:
                has_vid = "pixel_values_videos" in inputs
                print(f"[DEBUG] Will call generate() with pixel_values_videos present? {has_vid}")

            # Generation config
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )

            # Only add sampling params if do_sample=True (otherwise transformers may warn they are ignored)
            if do_sample:
                gen_kwargs.update(dict(
                    temperature=temperature,
                    top_p=top_p,
                ))

            # EOS if available
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = eos_id

            # IMPORTANT: pass the whole inputs dict to generate (includes pixel_values_videos)
            generated_ids = model.generate(**inputs, **gen_kwargs)

            # Decode
            decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Postprocess: remove prompt if it leaks into output
            cleaned: List[str] = []
            for s in decoded:
                s2 = s.strip()
                # common cleanup if model repeats prompt:
                if "Translation:" in s2:
                    s2 = s2.split("Translation:", 1)[-1].strip()
                cleaned.append(s2)

            # Save per item
            for vp, ref, pred in zip(videos_batch, texts_batch, cleaned):
                preds.append(pred)
                refs.append(ref)
                rec = {"video_path": vp, "ref": ref, "pred": pred}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1

                # -------------------- DEBUG: print some samples --------------------
                if n_done <= debug_first_n:
                    print("\n---------------- SAMPLE DEBUG ----------------")
                    print("VIDEO:", vp)
                    print("REF :", ref)
                    print("PRED:", pred)
                    print("---------------------------------------------\n")

                if n_done >= max_samples:
                    break

            pbar.set_postfix({"done": n_done})

    dt = time.time() - t0
    return {"n_samples": n_done, "seconds": dt, "preds": preds, "refs": refs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--out_jsonl", type=str, required=True)

    parser.add_argument("--stage1_dir", type=str, default=None, help="dir containing adapter.pt for stage1 epoch_best")
    parser.add_argument("--stage2_dir", type=str, default=None, help="dir containing adapter.pt for stage2 (epoch_best/epoch_latest/intra_latest)")

    parser.add_argument("--dataset_json", type=str, default=str(PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset_clean.json"))
    parser.add_argument("--root_dir", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--num_workers", type=int, default=2)

    # generation params
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--do_sample", action="store_true", help="enable sampling (otherwise greedy)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    # debug
    parser.add_argument("--debug_first_n", type=int, default=5, help="print first N predictions")
    args = parser.parse_args()

    out_jsonl = Path(args.out_jsonl)

    print("[INFO] Running eval:")
    print(f"  stage      = {args.stage}")
    print(f"  split      = {args.split}")
    print(f"  max_samples= {args.max_samples}")
    print(f"  stage1_dir = {args.stage1_dir}")
    print(f"  stage2_dir = {args.stage2_dir}")
    print(f"  out_jsonl  = {out_jsonl}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # -----------------------------------------------------------------
    # Load base + attach LoRA for chosen stage (your helper does this)
    # -----------------------------------------------------------------
    t0 = time.time()
    if args.stage == "stage1":
        model, processor = load_qwen3vl_lora(
            model_name=args.model_name,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
    else:
        # Stage2: you likely want to merge stage1 LoRA first (for your 2-stage setup)
        model, processor = load_qwen3vl_lora(
            model_name=args.model_name,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage2",
            stage1_adapter_dir=args.stage1_dir,
        )
    print(f"[TIMER] Total load_qwen3vl_lora() time: {time.time() - t0:.2f}s")

    # -----------------------------------------------------------------
    # Load Stage2 adapter weights (ONLY LoRA keys) if provided
    # -----------------------------------------------------------------
    if args.stage2_dir is not None:
        adapter_path = Path(args.stage2_dir) / "adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Stage2 adapter not found: {adapter_path}")
        missing, unexpected = load_adapter_into_model_only_lora(model, adapter_path, device=device)
        print(f"[CKPT] Stage2 adapter loaded: missing={missing}, unexpected={unexpected}")

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    ds = How2SignDataset(
        json_path=str(args.dataset_json),
        split=args.split,
        root_dir=str(args.root_dir),
        return_type="video",
    )
    print(f"[How2SignDataset] split={args.split} | num_samples={len(ds)}")
    print(f"[How2SignDataset] json_path={args.dataset_json}")
    print(f"[How2SignDataset] root_dir={args.root_dir}")

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=how2sign_collate_fn,
        pin_memory=(device == "cuda"),
    )

    # -----------------------------------------------------------------
    # Run generation + save jsonl
    # -----------------------------------------------------------------
    gen = run_generation(
        model=model,
        processor=processor,
        device=device,
        loader=loader,
        out_jsonl=out_jsonl,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        debug_first_n=args.debug_first_n,
    )

    print(f"[INFO] Generation done in {gen['seconds']:.1f}s for {gen['n_samples']} samples.")
    print(f"[INFO] Saved predictions to: {out_jsonl}")

    preds = gen["preds"]
    refs = gen["refs"]

    # -----------------------------------------------------------------
    # Metrics (BLEU1..4 + ROUGE-L)
    # -----------------------------------------------------------------
    bleu1 = corpus_bleu_n(preds, refs, n=1)
    bleu2 = corpus_bleu_n(preds, refs, n=2)
    bleu3 = corpus_bleu_n(preds, refs, n=3)
    bleu4 = corpus_bleu_n(preds, refs, n=4)
    rouge_l = corpus_rouge_l(preds, refs)

    metrics = {
        "split": args.split,
        "stage": args.stage,
        "n_samples": gen["n_samples"],
        "BLEU1": bleu1,
        "BLEU2": bleu2,
        "BLEU3": bleu3,
        "BLEU4": bleu4,
        "ROUGE_L_F1": rouge_l,
    }

    print("\n================ METRICS ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.6f}")
        else:
            print(f"{k:>12}: {v}")
    print("=========================================\n")

    # Save metrics next to jsonl
    metrics_path = out_jsonl.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()