#!/usr/bin/env python
"""
Utility per il dataset How2Sign:

1) Conta quanti video sono mancanti in ciascuno split.
2) Calcola statistiche sulle durate (media, max, ecc.).
3) Stampa un istogramma testuale delle durate e, se possibile,
   salva un grafico PNG.
4) Calcola quante clip superano varie soglie temporali (es. >10s, >20s, >30s).
5) Suggerisce una soglia automatica per considerare outlier.
6) Suggerisce valori ragionevoli per fps sampling e max_frames.

Si lancia semplicemente con:

    python scripts/check_how2sign_video.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config di base (modifica qui se cambi layout del repo)
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # root del repo
JSON_PATH = PROJECT_ROOT / "data/How2Sign/how2sign_dataset.json"
ROOT_DIR = PROJECT_ROOT

# Soglie per contare i "video lunghi" (in secondi)
THRESH_LIST = [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def load_how2sign_json(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON non trovato: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def check_missing_videos(
    splits: Dict[str, List[Dict[str, Any]]],
    root_dir: Path,
) -> Dict[str, List[str]]:
    """
    Ritorna un dict split_name -> lista di path relativi mancanti.
    """
    missing: Dict[str, List[str]] = {}

    for split_name, entries in splits.items():
        print(f"[CHECK] Controllo split='{split_name}' con {len(entries)} samples...")
        missing[split_name] = []

        for entry in tqdm(entries, desc=split_name):
            video_rel = Path(entry["video_path"])
            video_abs = (root_dir / video_rel).resolve()
            if not video_abs.exists():
                missing[split_name].append(str(video_rel))

    return missing

def clean_json_missing_videos(
    splits: Dict[str, List[Dict[str, Any]]],
    missing: Dict[str, List[str]],
    json_path: Path,
    backup: bool = True,
) -> None:
    """
    Rimuove dal JSON tutte le entry i cui video risultano mancanti.

    Args:
        splits:   dizionario split -> lista di entry (caricato dal JSON).
        missing:  dizionario split -> lista di path relativi mancanti (generato da check_missing_videos).
        json_path: path all'originale how2sign_dataset.json.
        backup:   se True, crea un file JSON di backup prima di sovrascrivere.

    Salva il JSON corretto nello stesso percorso.
    """
    total_removed = 0

    # Copia profonda dello struttura
    new_splits = {}
    for split_name, entries in splits.items():
        missing_set = set(missing.get(split_name, []))
        new_entries = []
        for entry in entries:
            if entry["video_path"] not in missing_set:
                new_entries.append(entry)
            else:
                total_removed += 1
        new_splits[split_name] = new_entries

    print(f"[CLEAN] Entry totali rimosse: {total_removed}")

    # Backup
    if backup:
        backup_path = json_path.with_suffix(".backup.json")
        json_path.rename(backup_path)
        print(f"[CLEAN] Backup salvato in: {backup_path}")

    # Scrittura JSON pulito
    cleaned = {"splits": new_splits}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"[CLEAN] JSON pulito salvato in: {json_path}")

def compute_stats_for_split(
    split_name: str,
    entries: List[Dict[str, Any]],
    root_dir: Path,
) -> Dict[str, Any]:
    """
    Calcola:
      - durata di ogni video
      - media
      - max + info
      - conteggio clip che superano le soglie in THRESH_LIST
    Restituisce anche la lista delle durate (per analisi globali).
    """
    durations: List[float] = []
    max_duration = -1.0
    max_info = None

    # Dizionario soglia -> numero di video con durata > soglia
    long_counts = {thr: 0 for thr in THRESH_LIST}

    print(f"[STATS] Split '{split_name}' ({len(entries)} samples)...")
    for entry in tqdm(entries, desc=f"stats-{split_name}"):
        video_rel = Path(entry["video_path"])
        video_abs = (root_dir / video_rel).resolve()

        cap = cv2.VideoCapture(str(video_abs))
        if not cap.isOpened():
            # se non si apre, saltiamo
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()

        if fps <= 0.0 or frame_count <= 0.0:
            continue

        duration = frame_count / fps
        durations.append(duration)

        # aggiorna conteggi per soglie
        for thr in THRESH_LIST:
            if duration > thr:
                long_counts[thr] += 1

        # aggiorna massimo
        if duration > max_duration:
            max_duration = duration
            max_info = {
                "split": split_name,
                "rel_path": str(video_rel),
                "abs_path": str(video_abs),
            }

    mean_duration = sum(durations) / len(durations) if durations else 0.0

    return {
        "durations": durations,
        "num_videos": len(durations),
        "mean_duration": mean_duration,
        "max_duration": max_duration,
        "max_info": max_info,
        "long_counts": long_counts,
    }


def print_text_histogram(durations: List[float], bins: int = 20) -> None:
    """
    Stampa a console un istogramma testuale delle durate.
    """
    if not durations:
        print("[HIST] Nessuna durata disponibile.")
        return

    arr = np.array(durations, dtype=np.float32)
    hist, bin_edges = np.histogram(arr, bins=bins)

    print("\n[HIST] Istogramma durate (secondi):")
    for i in range(len(hist)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar = "#" * min(50, int(50 * count / max(hist))) if hist.max() > 0 else ""
        print(f"  [{left:6.2f}, {right:6.2f})  {count:6d}  {bar}")


# def save_histogram_png(durations: List[float], out_path: Path) -> None:
#     """
#     Prova a salvare un istogramma come PNG.
#     Se matplotlib non è installato, stampa un warning e salta.
#     """
#     try:
#         import matplotlib.pyplot as plt  # type: ignore
#     except ImportError:
#         print("[HIST] matplotlib non installato, salto il salvataggio del grafico PNG.")
#         return

#     if not durations:
#         print("[HIST] Nessuna durata disponibile, niente PNG.")
#         return

#     arr = np.array(durations, dtype=np.float32)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.figure(figsize=(8, 4))
#     plt.hist(arr, bins=30)
#     plt.title("How2Sign clip duration distribution (seconds)")
#     plt.xlabel("Duration (s)")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()
#     print(f"[HIST] Istogramma salvato in: {out_path}")


def suggest_outlier_threshold_and_frames(durations: List[float]) -> None:
    """
    - Calcola una soglia per gli outlier via IQR.
    - Mostra alcune percentile (p50, p75, p90, p95, p99).
    - Suggerisce un fps e un max_frames ragionevoli, assumendo di voler
      avere ~32 frame per il 95-esimo percentile.
    """
    if not durations:
        print("[SUGGEST] Nessuna durata disponibile.")
        return

    arr = np.array(durations, dtype=np.float32)

    p50, p75, p90, p95, p99 = np.percentile(arr, [50, 75, 90, 95, 99])
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    iqr_thr = q3 + 1.5 * iqr

    print("\n[SUGGEST] Percentili durate (secondi):")
    print(f"  p50 (mediana): {p50:.2f}s")
    print(f"  p75:           {p75:.2f}s")
    print(f"  p90:           {p90:.2f}s")
    print(f"  p95:           {p95:.2f}s")
    print(f"  p99:           {p99:.2f}s")

    print("\n[SUGGEST] Soglia outlier (IQR rule):")
    print(f"  q1:      {q1:.2f}s")
    print(f"  q3:      {q3:.2f}s")
    print(f"  IQR:     {iqr:.2f}s")
    print(f"  thr_IQR: {iqr_thr:.2f}s  (q3 + 1.5*IQR)")

    n_out_iqr = int((arr > iqr_thr).sum())
    print(f"  -> Video oltre thr_IQR: {n_out_iqr} ({100 * n_out_iqr / len(arr):.2f}%)")

    # Suggerimento fps + max_frames:
    # vogliamo circa 32 frame alla durata del 95-esimo percentile
    target_frames_95 = 32
    if p95 > 0:
        fps_suggest = target_frames_95 / p95
    else:
        fps_suggest = 2.0  # fallback random

    # Clip di durata d avranno ~min(d * fps, max_frames_suggest) frame
    max_frames_suggest = target_frames_95

    print("\n[SUGGEST] Suggerimento per fps sampling e max_frames:")
    print(f"  Se vuoi circa {target_frames_95} frame per il 95-esimo percentile:")
    print(f"    -> fps suggeriti ≈ {fps_suggest:.2f} frame/sec")
    print(f"    -> max_frames suggerito: {max_frames_suggest} (clippando i video più lunghi)")

    # Calcoliamo quanti frame avrebbe ogni clip con questo schema (solo per info)
    est_frames = np.minimum(arr * fps_suggest, max_frames_suggest)
    mean_frames = est_frames.mean()
    p95_frames = np.percentile(est_frames, 95)
    p99_frames = np.percentile(est_frames, 99)

    print("\n[SUGGEST] Stima #frame con questo schema (fps, max_frames):")
    print(f"  - frame medi:   {mean_frames:.2f}")
    print(f"  - frame p95:    {p95_frames:.2f}")
    print(f"  - frame p99:    {p99_frames:.2f}")
    print("  (I video lunghi verranno tagliati a max_frames, quelli corti avranno meno frame.)")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    print(f"[INFO] JSON:     {JSON_PATH}")
    print(f"[INFO] root_dir: {ROOT_DIR}\n")

    data = load_how2sign_json(JSON_PATH)
    splits: Dict[str, List[Dict[str, Any]]] = data.get("splits", {})

    # 1) Check video mancanti
    missing = check_missing_videos(splits, ROOT_DIR)

    print("\n[REPORT] Video mancanti:")
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True, parents=True)

    total_missing = 0
    for split_name in ["train", "val", "test"]:
        split_missing = missing.get(split_name, [])

        print(f"  - {split_name}: {len(split_missing)}")
        total_missing += len(split_missing)

        # Salva la lista dei file mancanti
        out_file = outputs_dir / f"missing_{split_name}.txt"
        out_file.write_text("\n".join(split_missing), encoding="utf-8")

    print(f"  >>> Totale mancanti: {total_missing}")
    print(f"[INFO] Liste file mancanti salvate in: {outputs_dir}\n")
    
    # --------------------------------------------------------
    # OPTIONAL: pulizia automatica del JSON
    # Decommenta SOLO se vuoi sovrascrivere il JSON!!!
    # --------------------------------------------------------
    # if total_missing > 0:
    #     print("[CLEAN] Rimuovo dal JSON tutte le entry con video mancanti...")
    #     clean_json_missing_videos(splits, missing, JSON_PATH, backup=True)
    #     return  # interrompi qui perché il JSON è cambiato

    # 2) Statistiche durate per split
    print("[STATS] Calcolo durata media e video più lungo...")

    stats_train = compute_stats_for_split("train", splits.get("train", []), ROOT_DIR)
    stats_val = compute_stats_for_split("val", splits.get("val", []), ROOT_DIR)
    stats_test = compute_stats_for_split("test", splits.get("test", []), ROOT_DIR)

    # 3) Aggregazione globale
    all_durations: List[float] = (
        stats_train["durations"]
        + stats_val["durations"]
        + stats_test["durations"]
    )
    total_videos = len(all_durations)

    if total_videos > 0:
        global_mean = sum(all_durations) / total_videos
    else:
        global_mean = 0.0

    # max globale
    max_candidates = [
        (stats_train["max_duration"], stats_train["max_info"]),
        (stats_val["max_duration"], stats_val["max_info"]),
        (stats_test["max_duration"], stats_test["max_info"]),
    ]
    global_max, global_max_info = max(
        max_candidates, key=lambda x: (x[0] if x[0] is not None else -1)
    )

    print("\n[STATS] RISULTATI GLOBALI")
    print(f"  - Video analizzati: {total_videos}")
    print(f"  - Durata media: {global_mean:.2f}s ({global_mean/60:.2f} min)")

    print("  - Video che superano le soglie:")
    for thr in THRESH_LIST:
        c = (
            stats_train["long_counts"][thr]
            + stats_val["long_counts"][thr]
            + stats_test["long_counts"][thr]
        )
        perc = c / total_videos * 100 if total_videos > 0 else 0.0
        print(f"      > {thr:>4.1f}s: {c} video ({perc:.2f}%)")

    print(f"  - Video più lungo: {global_max:.2f}s ({global_max/60:.2f} min)")
    if global_max_info is not None:
        print("    Dettagli:")
        print(f"      split: {global_max_info['split']}")
        print(f"      rel path: {global_max_info['rel_path']}")
        print(f"      abs path: {global_max_info['abs_path']}")

    # 4) Istogramma testuale + PNG
    print_text_histogram(all_durations, bins=20)
    hist_png_path = PROJECT_ROOT / "outputs" / "how2sign_duration_hist.png"
    # save_histogram_png(all_durations, hist_png_path)

    # 5) Suggerimenti outlier + fps + max_frames
    suggest_outlier_threshold_and_frames(all_durations)


if __name__ == "__main__":
    main()