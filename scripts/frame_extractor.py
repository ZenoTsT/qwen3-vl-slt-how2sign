# ---------------------------------------------------------------- #
#  QUESTO SCRIPT SERVE PER ESTRARRE E MEMORIZZARE I FRAME DI OGNI  #
#  VIDEO IN MODO DA NON DOVERLO FARE SUCCESSIVAMENTE OGNI VOLTA    #
#  IN RUNTIME. NON UTILIZZATO PER ORA (TROPPO SPAZIO)              #
# ---------------------------------------------------------------- #

#!/usr/bin/env python
import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------
# Add project root to PYTHONPATH
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.video_io import extract_frames_from_video  # noqa: E402

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SPLITS = ["train", "val", "test"]

RAW_DIR_TEMPLATE = "data/How2Sign/{}_raw_videos"
FRAMES_DIR_TEMPLATE = "data/How2Sign/{}_frames"

N_FRAMES_TO_TAKE = None  # None = prendi tutti i frame

NUM_WORKERS = 6


# ---------------------------------------------------------------------
# Worker function (eseguita in parallelo)
# ---------------------------------------------------------------------
def process_single_video(task: Tuple[str, Path]):
    """
    task = (video_path, frames_root_dir)
    """
    video_path_str, frames_root = task
    video_path = Path(video_path_str)
    video_name = video_path.stem

    out_dir = frames_root / video_name

    # -------- Resume: se la cartella esiste già con dei frame, skippa ----------
    if out_dir.exists():
        existing_frames = list(out_dir.glob("frame_*.jpg"))
        if len(existing_frames) > 0:
            return (video_name, len(existing_frames), "SKIPPED_ALREADY_EXISTS")

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Estrazione dei frame ----------
    try:
        frames: List[Image.Image] = extract_frames_from_video(
            video_path=str(video_path),
            n_frames_to_take=N_FRAMES_TO_TAKE,
            strategy="uniform",
        )
    except Exception as e:
        return (video_name, 0, f"ERROR extracting frames: {e}")

    # -------- Salvataggio frame ----------
    saved = 0
    for i, frame in enumerate(frames):
        out_path = out_dir / f"frame_{i:05d}.jpg"
        try:
            frame.save(out_path, quality=95)
            saved += 1
        except Exception as e:
            return (video_name, saved, f"ERROR saving frame {i}: {e}")

    return (video_name, saved, "OK")


# ---------------------------------------------------------------------
# Processa un intero split (train/val/test)
# ---------------------------------------------------------------------
def extract_frames_for_split(split: str):
    raw_dir = ROOT_DIR / RAW_DIR_TEMPLATE.format(split)
    frames_dir = ROOT_DIR / FRAMES_DIR_TEMPLATE.format(split)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(raw_dir.glob("*.mp4"))
    print(f"\n[INFO] Split '{split}': {len(video_files)} video trovati in {raw_dir}")

    if len(video_files) == 0:
        return

    tasks = [(str(v), frames_dir) for v in video_files]

    results = []
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for r in tqdm(
            pool.imap_unordered(process_single_video, tasks),
            total=len(tasks),
            desc=f"Extracting {split}",
        ):
            results.append(r)

    ok = sum(1 for _, _, status in results if status == "OK")
    skipped = sum(1 for _, _, status in results if status == "SKIPPED_ALREADY_EXISTS")
    errors = [(name, msg) for (name, _, msg) in results
              if msg not in ("OK", "SKIPPED_ALREADY_EXISTS")]

    print(f"[INFO] {split}: {ok} OK, {skipped} skipped, {len(errors)} errors.")
    if errors:
        print(f"[WARN] Video con errori ({len(errors)}):")
        for name, msg in errors[:20]:
            print(f"   - {name}: {msg}")
        if len(errors) > 20:
            print(f"   ... (+{len(errors) - 20} altri)")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=====================================================")
    print("      HOW2SIGN — FAST MULTI-PROCESS FRAME EXTRACTOR")
    print("=====================================================\n")

    print(f"[INFO] Project root: {ROOT_DIR}")
    print(f"[INFO] Using {NUM_WORKERS} CPU workers\n")

    for split in SPLITS:
        extract_frames_for_split(split)

    print("\n[INFO] Frame extraction finished for all splits.")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    main()