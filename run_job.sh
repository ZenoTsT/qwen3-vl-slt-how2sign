#!/bin/bash
#SBATCH --job-name=qwen3vl_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --account=tesi_ztesta
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# Carico anaconda
module load anaconda3/2023.09-0-none-none

# Attivo l'env
source activate qwen3vl_env

# Mi sposto nella cartella del progetto
cd /homes/ztesta/qwen3-vl-slt-how2sign

# PYTHONPATH
export PYTHONPATH=/homes/ztesta/qwen3-vl-slt-how2sign:$PYTHONPATH

# Logs dir
mkdir -p logs
mkdir -p outputs/qwen3vl_lora_how2sign/logs

# ------------------------------
# Config eval
# ------------------------------
STAGE="stage2"
SPLIT="val"
MAX_SAMPLES=200
MAX_NEW_TOKENS=64

STAGE1_DIR="outputs/qwen3vl_lora_how2sign/checkpoints/stage1/epoch_best"
STAGE2_DIR="outputs/qwen3vl_lora_how2sign/checkpoints/stage2/intra_latest"

OUT_JSONL="outputs/qwen3vl_lora_how2sign/logs/eval_${STAGE}_${SPLIT}_${SLURM_JOB_ID}.jsonl"

# (Consigliato) log più puliti / meno warning
export TOKENIZERS_PARALLELISM=false

echo "[INFO] Running eval:"
echo "  stage      = ${STAGE}"
echo "  split      = ${SPLIT}"
echo "  max_samples= ${MAX_SAMPLES}"
echo "  stage1_dir = ${STAGE1_DIR}"
echo "  stage2_dir = ${STAGE2_DIR}"
echo "  out_jsonl  = ${OUT_JSONL}"

python scripts/eval_checkpoints.py \
  --stage "${STAGE}" \
  --stage1_dir "${STAGE1_DIR}" \
  --stage2_dir "${STAGE2_DIR}" \
  --split "${SPLIT}" \
  --batch_size 1 \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --out_jsonl "${OUT_JSONL}"