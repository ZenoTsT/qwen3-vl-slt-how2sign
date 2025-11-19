import os
import sys
from pathlib import Path
from typing import List

from PIL import Image

# ----------------------------------------------------------
# scripts/ is one level below the root, so we go up once.
# ----------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.utils.video_io import extract_frames_from_video


def main():

    # ----------------------------------------------------------
    # 1) Path to test video
    # ----------------------------------------------------------
    video_path = "tests/assets/_fZbAxSSbX4_0-5-rgb_front.mp4"

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"[ERROR] Video not found: {video_path}.\n"
            f"Check tests/assets/ for the correct filename."
        )

    # ----------------------------------------------------------
    # 2) Extract ALL frames (n_frames_to_take=None)
    # ----------------------------------------------------------
    print(f"[INFO] Extracting ALL frames from: {video_path}")

    frames: List[Image.Image] = extract_frames_from_video(
        video_path=video_path,
        n_frames_to_take=None,     # take all frames
    )

    print(f"[INFO] extract_frames_from_video() returned {len(frames)} frames.")

    if len(frames) == 0:
        print("[WARN] No frames extracted!")
        return

    # ----------------------------------------------------------
    # 3) Save the first 10 frames for visual inspection
    # ----------------------------------------------------------
    out_dir = Path("tests/debug_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_to_save = min(47, len(frames))
    for i in range(n_to_save):
        out_path = out_dir / f"frame_{i:03d}.jpg"
        frames[i].save(out_path)
        print(f"[INFO] Saved {out_path}")

    print("[INFO] Done. Check tests/debug_frames/ for extracted frames.")


if __name__ == "__main__":
    main()