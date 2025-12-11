#!/usr/bin/env python
import json
import subprocess
from pathlib import Path

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TARGET_SIZE = 384           # lato finale del quadrato
PRINT_EVERY = 500           # stampa ogni N video

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATASET_JSON = PROJECT_ROOT / "data/How2Sign/how2sign_dataset.json"

OUTPUT_ROOT = PROJECT_ROOT / "data/How2Sign_resized"


def run_ffmpeg(input_path: Path, output_path: Path, target_size: int = 384):
    vf_filter = (
        f"scale=-2:{target_size},"
        f"crop={target_size}:{target_size}:(in_w-{target_size})/2:(in_h-{target_size})/2"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "copy",
        str(output_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed for {input_path}")
        print(result.stderr.decode("utf-8", errors="ignore"))
    # se vuoi meno spam: togli l'else


def main():
    print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[INFO] DATASET_JSON: {DATASET_JSON}")

    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"JSON non trovato: {DATASET_JSON}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] OUTPUT_ROOT: {OUTPUT_ROOT}")

    # ------------------------------------------------------------
    # Leggo JSON
    # ------------------------------------------------------------
    with DATASET_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data.get("splits", {})
    split_names = ["train", "val", "test"]

    # Conta totale video
    total_expected = sum(len(splits.get(s, [])) for s in split_names)
    print(f"[INFO] Totale video da processare: {total_expected}")

    processed = 0

    # ------------------------------------------------------------
    # Loop su tutti gli split
    # ------------------------------------------------------------
    for split_name in split_names:
        entries = splits.get(split_name, [])
        print(f"\n[INFO] Split '{split_name}': {len(entries)} samples")

        for entry in entries:
            video_rel = Path(entry["video_path"])
            video_abs = (PROJECT_ROOT / video_rel).resolve()

            if not video_abs.exists():
                print(f"[WARN] Video mancante: {video_abs}")
                continue

            try:
                rel_to_how2sign = video_rel.relative_to("data/How2Sign")
            except ValueError:
                rel_to_how2sign = video_rel

            out_path = (OUTPUT_ROOT / rel_to_how2sign).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if not out_path.exists():
                run_ffmpeg(video_abs, out_path, target_size=TARGET_SIZE)

            processed += 1

            # ---- PRINT PERIODICO ----
            if processed % PRINT_EVERY == 0:
                percentage = (processed / total_expected) * 100
                print(f"[STATUS] Processati {processed}/{total_expected} "
                      f"({percentage:.2f}%)")

    print(f"\n[DONE] Preprocessing completato. Video processati: {processed}/{total_expected}")


if __name__ == "__main__":
    main()