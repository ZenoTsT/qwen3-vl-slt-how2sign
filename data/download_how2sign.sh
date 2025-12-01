#!/bin/bash
# Simple How2Sign downloader for:
#  - sentence-level RGB front clips
#  - re-aligned English translations
#
# Final structure:
#   data/How2Sign/sentence_level/{train,val,test}/rgb_front/*.mp4
#   data/How2Sign/sentence_level/{train,val,test}/text/*.csv
#
# Run this script from the `data/` folder of the project:
#   cd data
#   ./download_how2sign.sh

set -euo pipefail

# -------- CONFIG -------- #

# Base folder where How2Sign will live (relative to this script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/How2Sign"

# Google Drive IDs (from original How2Sign script)
ID_TRAIN_CLIPS="1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW"
ID_VAL_CLIPS="1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO"
ID_TEST_CLIPS="1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG"

ID_TRAIN_REALIGNED="1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok"
ID_VAL_REALIGNED="1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI"
ID_TEST_REALIGNED="1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu"

# ------------------------ #

echo "[INFO] Base How2Sign folder: ${BASE_DIR}"

# Check gdown is installed
if ! command -v gdown &>/dev/null; then
  echo "[ERROR] gdown not found. Install it in your env, e.g.:"
  echo "        pip install gdown"
  exit 1
fi

# Create directory structure
echo "[INFO] Creating directory structure..."
mkdir -p "${BASE_DIR}/sentence_level/train/rgb_front"
mkdir -p "${BASE_DIR}/sentence_level/val/rgb_front"
mkdir -p "${BASE_DIR}/sentence_level/test/rgb_front"

mkdir -p "${BASE_DIR}/sentence_level/train/text"
mkdir -p "${BASE_DIR}/sentence_level/val/text"
mkdir -p "${BASE_DIR}/sentence_level/test/text"

echo "[INFO] Downloading RGB front clips (sentence-level)..."

# Paths for temporary zips
TRAIN_ZIP="${BASE_DIR}/train_rgb_front_clips.zip"
VAL_ZIP="${BASE_DIR}/val_rgb_front_clips.zip"
TEST_ZIP="${BASE_DIR}/test_rgb_front_clips.zip"

echo "  - Train clips..."
gdown "https://drive.google.com/uc?id=${ID_TRAIN_CLIPS}" -O "${TRAIN_ZIP}"
echo "  - Val clips..."
gdown "https://drive.google.com/uc?id=${ID_VAL_CLIPS}" -O "${VAL_ZIP}"
echo "  - Test clips..."
gdown "https://drive.google.com/uc?id=${ID_TEST_CLIPS}" -O "${TEST_ZIP}"

echo "[INFO] Unzipping RGB front clips..."
unzip -q "${TRAIN_ZIP}" -d "${BASE_DIR}/sentence_level/train/rgb_front"
unzip -q "${VAL_ZIP}"   -d "${BASE_DIR}/sentence_level/val/rgb_front"
unzip -q "${TEST_ZIP}"  -d "${BASE_DIR}/sentence_level/test/rgb_front"

rm -f "${TRAIN_ZIP}" "${VAL_ZIP}" "${TEST_ZIP}"

echo "[INFO] Downloading re-aligned English translations..."

TRAIN_CSV="${BASE_DIR}/sentence_level/train/text/how2sign_realigned_train.csv"
VAL_CSV="${BASE_DIR}/sentence_level/val/text/how2sign_realigned_val.csv"
TEST_CSV="${BASE_DIR}/sentence_level/test/text/how2sign_realigned_test.csv"

echo "  - Train CSV..."
gdown "https://drive.google.com/uc?id=${ID_TRAIN_REALIGNED}" -O "${TRAIN_CSV}"
echo "  - Val CSV..."
gdown "https://drive.google.com/uc?id=${ID_VAL_REALIGNED}" -O "${VAL_CSV}"
echo "  - Test CSV..."
gdown "https://drive.google.com/uc?id=${ID_TEST_REALIGNED}" -O "${TEST_CSV}"

echo "[INFO] Done. Final structure:"
echo "  ${BASE_DIR}/sentence_level/train/rgb_front"
echo "  ${BASE_DIR}/sentence_level/val/rgb_front"
echo "  ${BASE_DIR}/sentence_level/test/rgb_front"
echo "  ${BASE_DIR}/sentence_level/{train,val,test}/text"