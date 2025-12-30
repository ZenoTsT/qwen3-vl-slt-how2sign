#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

# --------------------------
# PYTHONPATH: root progetto
# --------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn
from src.models.qwen3vl_lora import load_qwen3vl_lora

# --------------------------
# Prompt (uguale al train)
# --------------------------
def build_instruction_prompt() -> str:
    return (
        "You are a sign language translation model. "
        "Given the following sign language video <|video_pad|>, "
        "translate it into English.\n\n"
        "Answer with the English sentence only.\n\n"
        "Translation:"
    )

# --------------------------
# Metriche: BLEU/ROUGE
# --------------------------
def _ensure_metrics_libs():
    """
    Prova a importare sacrebleu e rouge_score.
    Se mancano, stampa istruzioni.
    """
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

def compute_bleu_1_4(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu

    # sacrebleu vuole refs come lista-di-liste: [refs]
    refs_wrapped = [refs]

    def bleu_with_order(n: int) -> float:
        # pesi: BLEU-n
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        bleu = sacrebleu.metrics.BLEU(
            smooth_method="exp",
            smooth_value=0.0,
            max_ngram_order=4,
            effective_order=True,
            tokenize="13a",
            lowercase=False,
            force=False,
            use_effective_order=True,
            # sacrebleu BLEU usa weights se passate in compute? no, quindi usiamo corpus_bleu con smooth + weights manualmente:
        )
        # workaround pulito: usare corpus_bleu direttamente (supporta smooth_method e ngram_order massimo)
        # ma BLEU-n non è “standard” in sacrebleu. Quindi calcoliamo BLEU classico e ricaviamo BLEU-n via ngram precisions:
        # -> più semplice: usiamo sacrebleu.corpus_bleu con max_ngram_order=n
        score = sacrebleu.corpus_bleu(
            preds,
            refs_wrapped,
            smooth_method="exp",
            smooth_value=0.0,
            tokenize="13a",
            lowercase=False,
            force=False,
            use_effective_order=True,
            effective_order=True,
            max_ngram_order=n,
        ).score
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

    r1_f, r2_f, rL_f = 0.0, 0.0, 0.0
    n = len(preds)

    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)  # (ref, pred)
        r1_f += scores["rouge1"].fmeasure
        r2_f += scores["rouge2"].fmeasure
        rL_f += scores["rougeL"].fmeasure

    if n == 0:
        return {"ROUGE1_F": float("nan"), "ROUGE2_F": float("nan"), "ROUGEL_F": float("nan")}

    return {
        "ROUGE1_F": float(r1_f / n),
        "ROUGE2_F": float(r2_f / n),
        "ROUGEL_F": float(rL_f / n),
    }

# --------------------------
# Checkpoint loader (adapter.pt)
# --------------------------
def load_adapter_weights_into_model(model: torch.nn.Module, adapter_path: Path, device: str) -> None:
    """
    Carica un adapter.pt (dict di tensori) e lo fonde nello state_dict del modello.
    Funziona per stage1 e stage2 perché i nomi dei parametri LoRA coincidono con quelli del modello.
    """
    adapter_state = torch.load(adapter_path, map_location="cpu")
    model_state = model.state_dict()

    loaded, skipped = 0, 0
    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k] = v.to(device)
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state, strict=False)
    print(f"[CKPT] Loaded adapter weights: loaded={loaded}, skipped={skipped} from {adapter_path}")

# --------------------------
# Generazione
# --------------------------
@torch.no_grad()
def generate_batch(model, processor, device: str, video_paths: List[str], max_new_tokens: int) -> List[str]:
    prompt = build_instruction_prompt()
    texts = [prompt] * len(video_paths)

    inputs = processor(
        videos=video_paths,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs.get("input_ids", None)
    in_len = input_ids.shape[1] if input_ids is not None else None

    # generate
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # prendiamo SOLO i token generati dopo il prompt
    if in_len is not None and gen_ids.shape[1] > in_len:
        gen_only = gen_ids[:, in_len:]
    else:
        gen_only = gen_ids

    preds = processor.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    preds = [p.strip() for p in preds]
    return preds

# --------------------------
# Main eval
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--json_path", default=str(PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset_clean.json"))
    parser.add_argument("--root_dir", default=str(PROJECT_ROOT))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])

    parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    parser.add_argument("--stage1_dir", default=str(PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign/checkpoints/stage1/epoch_best"))
    parser.add_argument("--stage2_dir", default=str(PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign/checkpoints/stage2/intra_latest"))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out_jsonl", default=None)
    args = parser.parse_args()

    _ensure_metrics_libs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] stage={args.stage}")
    print(f"[INFO] split={args.split} | max_samples={args.max_samples}")

    # --------------------------
    # 1) Load model + processor
    # --------------------------
    if args.stage == "stage1":
        model, processor = load_qwen3vl_lora(
            model_name=args.model_name,
            device=device,
            stage="stage1",
        )
        adapter_path = Path(args.stage1_dir) / "adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Stage1 adapter not found: {adapter_path}")
        load_adapter_weights_into_model(model, adapter_path, device)

    else:
        # stage2: merge stage1 nel base model, poi attach LoRA stage2, poi carica adapter stage2
        model, processor = load_qwen3vl_lora(
            model_name=args.model_name,
            device=device,
            stage="stage2",
            stage1_adapter_dir=args.stage1_dir,  # directory che contiene adapter.pt
        )
        adapter_path = Path(args.stage2_dir) / "adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Stage2 adapter not found: {adapter_path}")
        load_adapter_weights_into_model(model, adapter_path, device)

    model.eval()

    # --------------------------
    # 2) Dataset + DataLoader
    # --------------------------
    ds = How2SignDataset(
        json_path=args.json_path,
        split=args.split,
        root_dir=args.root_dir,
        return_type="video",
    )

    # subset
    n = min(len(ds), args.max_samples)
    indices = list(range(n))

    # dataloader manuale (semplice, no DDP)
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=how2sign_collate_fn,
        pin_memory=True,
    )

    # output file
    if args.out_jsonl is None:
        tag = f"{args.stage}_{args.split}_{n}"
        out_path = PROJECT_ROOT / "outputs" / "qwen3vl_lora_how2sign" / "logs" / f"predictions_{tag}.jsonl"
    else:
        out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds_all, refs_all = [], []
    t0 = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            video_paths = batch["videos"]  # list[str]
            refs = batch["texts"]          # list[str]

            # (opzionale) se ci sono path relativi, li rendiamo assoluti rispetto a root
            # ma nel tuo JSON sono già relativi al repo -> processor accetta anche relativi se cwd è root
            preds = generate_batch(model, processor, device, video_paths, args.max_new_tokens)

            for vp, r, p in zip(video_paths, refs, preds):
                preds_all.append(p)
                refs_all.append(r)
                f.write(json.dumps({"video_path": vp, "ref": r, "pred": p}, ensure_ascii=False) + "\n")

    t1 = time.perf_counter()
    print(f"[INFO] Generation done in {t1 - t0:.1f}s for {len(preds_all)} samples.")
    print(f"[INFO] Saved predictions to: {out_path}")

    # --------------------------
    # 3) Metrics
    # --------------------------
    bleu = compute_bleu_1_4(preds_all, refs_all)
    rouge = compute_rouge(preds_all, refs_all)

    metrics = {**bleu, **rouge, "num_samples": len(preds_all)}
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:10s}: {v:.4f}")
        else:
            print(f"{k:10s}: {v}")

    # salva anche metrics json
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[INFO] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()