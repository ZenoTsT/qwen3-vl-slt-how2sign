# scripts/build_how2sign_json.py

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

# ----------------------------------------------------------
# Imposto ROOT_DIR = root del progetto
# ----------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent  # qwen3-vl-slt-how2sign/
DATA_DIR = ROOT_DIR / "data" / "How2Sign_resized"

OUTPUT_JSON = DATA_DIR / "how2sign_dataset.json"

SPLITS = ["train", "val", "test"]

# Molto probabile: file TSV (tab-separated)
CSV_DELIMITER = "\t"


def parse_float(value: str):
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_split_entries(split: str) -> List[Dict[str, Any]]:
    """
    Legge il CSV realigned di uno split (train/val/test) e costruisce
    una lista di esempi con:
      - path al video (relativo alla root del repo)
      - testo target (SENTENCE)
      - metadati utili
    """
    csv_path = DATA_DIR / f"how2sign_realigned_{split}.csv"
    videos_dir = DATA_DIR / f"{split}_raw_videos"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for split '{split}': {csv_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found for split '{split}': {videos_dir}")

    print(f"[INFO] Building entries for split '{split}'")
    print(f"       CSV:    {csv_path}")
    print(f"       videos: {videos_dir}")

    entries: List[Dict[str, Any]] = []

    missing_videos = 0
    total_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        # 👉 QUI: delimiter=CSV_DELIMITER (tab)
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)

        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in CSV: {csv_path}")

        # Normalizzazione header:
        # - strip spazi
        # - rimuovi BOM (\ufeff)
        # - upper per essere case-insensitive
        def norm(name: str) -> str:
            return name.strip().lstrip("\ufeff").upper()

        norm_to_orig: Dict[str, str] = {norm(name): name for name in reader.fieldnames}

        # Helper per leggere un campo in modo robusto
        def get_field(row: Dict[str, str], key_norm: str, default: str = "") -> str:
            orig = norm_to_orig.get(key_norm)
            if orig is None:
                return default
            return row.get(orig, default)

        for row in reader:
            total_rows += 1

            # Campi principali (usiamo i nomi normalizzati)
            video_id = get_field(row, "VIDEO_ID").strip()
            video_name = get_field(row, "VIDEO_NAME").strip()
            sentence_id = get_field(row, "SENTENCE_ID").strip()
            sentence_name = get_field(row, "SENTENCE_NAME").strip()
            sentence_text = get_field(row, "SENTENCE").strip()

            start_realigned = parse_float(get_field(row, "START_REALIGNED"))
            end_realigned = parse_float(get_field(row, "END_REALIGNED"))

            if sentence_name == "":
                # Riga malformata, la skippiamo
                continue

            # Il file video è SENTENCE_NAME.mp4 dentro {split}_raw_videos
            clip_filename = f"{sentence_name}.mp4"
            clip_path = videos_dir / clip_filename

            if not clip_path.exists():
                missing_videos += 1
                # Debug opzionale:
                # print(f"[WARN] Missing video file for SENTENCE_NAME={sentence_name}")
                continue

            # Salviamo il path relativo alla root del repo (più portabile)
            rel_video_path = clip_path.relative_to(ROOT_DIR)

            sample = {
                "split": split,
                "video_path": str(rel_video_path),  # es. "data/How2Sign/train_raw_videos/xxx.mp4"
                "sentence": sentence_text,

                # Metadati extra
                "video_id": video_id,
                "video_name": video_name,
                "sentence_id": sentence_id,
                "sentence_name": sentence_name,
                "start_realigned": start_realigned,
                "end_realigned": end_realigned,
            }
            entries.append(sample)

    print(f"[INFO] Split '{split}': {len(entries)} esempi validi (su {total_rows} righe CSV)")
    if missing_videos > 0:
        print(f"[WARN] Split '{split}': {missing_videos} righe con video mancante (skippate)")

    return entries


def main():
    print("===============================================")
    print("   HOW2SIGN — JSON BUILDER (clip-level)")
    print("===============================================\n")

    print(f"[INFO] Project root: {ROOT_DIR}")
    print(f"[INFO] Data dir:     {DATA_DIR}\n")

    splits_dict: Dict[str, List[Dict[str, Any]]] = {}

    for split in SPLITS:
        entries = build_split_entries(split)
        splits_dict[split] = entries

    num_samples_total = sum(len(v) for v in splits_dict.values())

    dataset_dict = {
        "meta": {
            "dataset": "How2Sign",
            "version": "realigned",
            "description": "Clip-level How2Sign dataset (RGB front, re-aligned captions).",
            "root_dir": str(ROOT_DIR),
            "num_samples_total": num_samples_total,
            "num_samples_per_split": {split: len(splits_dict[split]) for split in SPLITS},
        },
        "splits": splits_dict,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    print("\n[INFO] JSON salvato in:")
    print(f"       {OUTPUT_JSON}")
    print(f"[INFO] Numero totale esempi: {num_samples_total}")


if __name__ == "__main__":
    main()