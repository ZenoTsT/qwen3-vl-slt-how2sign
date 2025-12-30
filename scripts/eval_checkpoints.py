#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn
from src.models.qwen3vl_lora import load_qwen3vl_lora


# ---------------------------------------------------------------------
# HARD-CODED CONFIG (NO PARAMS)
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# Always test split (as you requested)
SPLIT = "test"

# Always stage2 in your current setup; change if needed
STAGE = "stage2"  # "stage1" or "stage2"

# Max samples for quick debug
MAX_SAMPLES = 20

# Generation
MAX_NEW_TOKENS = 48
DO_SAMPLE = False            # greedy by default (stable)
TEMPERATURE = 0.7            # used only if DO_SAMPLE=True
TOP_P = 0.9                  # used only if DO_SAMPLE=True
REPETITION_PENALTY = 1.2

# Debug prints
DEBUG_FIRST_N = 5

# Paths
DATASET_JSON = PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset_clean.json"
ROOT_DIR = PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign"
OUT_JSONL = OUTPUT_DIR / "logs" / f"eval_{STAGE}_{SPLIT}_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"

# Stage adapters
STAGE1_DIR = OUTPUT_DIR / "checkpoints" / "stage1" / "epoch_best"
STAGE2_DIR = OUTPUT_DIR / "checkpoints" / "stage2" / "intra_latest"

# Loader
NUM_WORKERS = 2
BATCH_SIZE = 1  # fixed; no CLI


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
# Metrics (NO sacrebleu) -> BLEU1..4 + ROUGE-L
# ---------------------------------------------------------------------
def _tokenize(s: str) -> List[str]:
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
    cand_counts = _ngram_counts(candidate, n)
    if not cand_counts:
        return (0, 0)

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
    best = ref_lens[0]
    best_dist = abs(best - cand_len)
    for rl in ref_lens[1:]:
        dist = abs(rl - cand_len)
        if dist < best_dist or (dist == best_dist and rl < best):
            best = rl
            best_dist = dist
    return best


def corpus_bleu_n(preds: List[str], refs: List[str], n: int, smoothing_eps: float = 1e-9) -> float:
    assert len(preds) == len(refs)
    if len(preds) == 0:
        return float("nan")

    import math

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

    if cand_len_sum == 0:
        return 0.0

    log_p_sum = 0.0
    for k in range(n):
        if p_den[k] == 0:
            p = smoothing_eps
        else:
            p = p_num[k] / p_den[k]
            if p == 0.0:
                p = smoothing_eps
        log_p_sum += math.log(p)

    geo_mean = math.exp(log_p_sum / n)

    if cand_len_sum > ref_len_sum:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len_sum / max(cand_len_sum, 1)))

    return float(bp * geo_mean)


def _lcs_length(a: List[str], b: List[str]) -> int:
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
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
        f1s.append(f1)

    return float(sum(f1s) / len(f1s))


# ---------------------------------------------------------------------
# Adapter loading (only lora_ keys)
# ---------------------------------------------------------------------
def load_adapter_into_model_only_lora(model, adapter_path: Path, device: str) -> Tuple[int, int]:
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
# Generation
# ---------------------------------------------------------------------
@torch.no_grad()
def run_generation(model, processor, device: str, loader: DataLoader) -> Dict[str, Any]:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    preds: List[str] = []
    refs: List[str] = []

    model.eval()

    eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)

    n_done = 0
    t0 = time.time()

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        pbar = tqdm(
            loader,
            total=min(MAX_SAMPLES, len(loader.dataset)),
            desc="Evaluating",
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            if n_done >= MAX_SAMPLES:
                break

            videos_batch: List[str] = batch["videos"]
            texts_batch: List[str] = batch["texts"]

            # ------------------------------------------------------------
            # Build prompts using Qwen-VL chat template (video + instruction)
            # ------------------------------------------------------------
            instruction = build_instruction_prompt()
            chat_texts: List[str] = []

            if hasattr(processor, "apply_chat_template"):
                for vp in videos_batch:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": vp},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]
                    txt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,  # adds assistant slot
                    )
                    chat_texts.append(txt)
            else:
                # fallback (less ideal)
                chat_texts = [instruction] * len(videos_batch)

            # DEBUG: show final prompt once
            if batch_idx == 0:
                print("\n================ DEBUG: CHAT TEMPLATE (FIRST SAMPLE) ================")
                print(chat_texts[0])
                print("=====================================================================\n")

            # ------------------------------------------------------------
            # Processor: pass BOTH videos and chat-text
            # ------------------------------------------------------------
            inputs = processor(
                videos=videos_batch,
                text=chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                # Se vuoi forzare frames/fps per debug, decommenta:
                # num_frames=32,
                # fps=2.0,
            )

            # DEBUG: show processor outputs once
            if batch_idx == 0:
                print("\n================ DEBUG: PROCESSOR OUTPUT (BATCH 0) ================")
                print("processor keys:", list(inputs.keys()))
                for k, v in inputs.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype',None)}")
                if "pixel_values_videos" not in inputs:
                    print(
                        "[WARN] 'pixel_values_videos' NOT found in processor output. "
                        "This may mean the video is not being used."
                    )
                print("===================================================================\n")

            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            if batch_idx == 0:
                print(f"[DEBUG] generate() called with pixel_values_videos present? {'pixel_values_videos' in inputs}")

            # ------------------------------------------------------------
            # Generate
            # ------------------------------------------------------------
            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
            )
            if DO_SAMPLE:
                gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P))
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = eos_id

            generated_ids = model.generate(**inputs, **gen_kwargs)
            decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # ------------------------------------------------------------
            # Clean: keep ONLY assistant answer, robustly
            # ------------------------------------------------------------
            cleaned: List[str] = []
            for s in decoded:
                s2 = s.strip()

                # If the old marker exists, cut after it
                if "Translation:" in s2:
                    s2 = s2.split("Translation:", 1)[-1].strip()

                # If chat markers remain, take the tail
                for marker in ["assistant\n", "assistant:", "Assistant:", "ASSISTANT:"]:
                    if marker in s2:
                        s2 = s2.split(marker, 1)[-1].strip()

                cleaned.append(s2)

            # ------------------------------------------------------------
            # Save per-sample
            # ------------------------------------------------------------
            for vp, ref, pred in zip(videos_batch, texts_batch, cleaned):
                preds.append(pred)
                refs.append(ref)
                f.write(
                    json.dumps({"video_path": vp, "ref": ref, "pred": pred}, ensure_ascii=False) + "\n"
                )
                n_done += 1

                if n_done <= DEBUG_FIRST_N:
                    print("\n---------------- SAMPLE DEBUG ----------------")
                    print("VIDEO:", vp)
                    print("REF :", ref)
                    print("PRED:", pred)
                    print("---------------------------------------------\n")

                if n_done >= MAX_SAMPLES:
                    break

            pbar.set_postfix({"done": n_done})

    dt = time.time() - t0
    return {"n_samples": n_done, "seconds": dt, "preds": preds, "refs": refs}


def main():
    # IMPORTANT: ignore any CLI args to avoid failures from extra flags in wrappers
    # (we just don't parse them at all)

    print("[INFO] Running eval (NO PARAMS):")
    print(f"  stage      = {STAGE}")
    print(f"  split      = {SPLIT}")
    print(f"  max_samples= {MAX_SAMPLES}")
    print(f"  stage1_dir = {STAGE1_DIR}")
    print(f"  stage2_dir = {STAGE2_DIR}")
    print(f"  out_jsonl  = {OUT_JSONL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Load model + processor
    t0 = time.time()
    if STAGE == "stage1":
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
    else:
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage2",
            stage1_adapter_dir=str(STAGE1_DIR),
        )
    print(f"[TIMER] load_qwen3vl_lora() time: {time.time() - t0:.2f}s")

    # Load stage2 adapter (only lora_ weights)
    if STAGE2_DIR is not None:
        adapter_path = Path(STAGE2_DIR) / "adapter.pt"
        if adapter_path.exists():
            missing, unexpected = load_adapter_into_model_only_lora(model, adapter_path, device=device)
            print(f"[CKPT] Stage2 adapter loaded: missing={missing}, unexpected={unexpected}")
        else:
            print(f"[WARN] Stage2 adapter not found at: {adapter_path} (continuing without stage2 adapter)")

    # Dataset (forced test)
    ds = How2SignDataset(
        json_path=str(DATASET_JSON),
        split=SPLIT,
        root_dir=str(ROOT_DIR),
        return_type="video",
    )
    print(f"[How2SignDataset] split={SPLIT} | num_samples={len(ds)}")
    print(f"[How2SignDataset] json_path={DATASET_JSON}")
    print(f"[How2SignDataset] root_dir={ROOT_DIR}")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=how2sign_collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Generate
    gen = run_generation(model, processor, device=device, loader=loader)
    print(f"[INFO] Generation done in {gen['seconds']:.1f}s for {gen['n_samples']} samples.")
    print(f"[INFO] Saved predictions to: {OUT_JSONL}")

    preds = gen["preds"]
    refs = gen["refs"]

    # Metrics
    bleu1 = corpus_bleu_n(preds, refs, n=1)
    bleu2 = corpus_bleu_n(preds, refs, n=2)
    bleu3 = corpus_bleu_n(preds, refs, n=3)
    bleu4 = corpus_bleu_n(preds, refs, n=4)
    rouge_l = corpus_rouge_l(preds, refs)

    metrics = {
        "split": SPLIT,
        "stage": STAGE,
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

    metrics_path = OUT_JSONL.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()