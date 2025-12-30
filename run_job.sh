#!/bin/bash
#SBATCH --job-name=qwen3vl_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --account=tesi_ztesta
#SBATCH --partition=all_usr_prod
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

# (Consigliato) log più puliti / meno warning
export TOKENIZERS_PARALLELISM=false

# Se vuoi silenziare un po' transformers:
# export TRANSFORMERS_VERBOSITY=error

echo "[INFO] Running eval (NO PARAMS)."
echo "[INFO] Script: scripts/eval_checkpoints.py"
echo "[INFO] NOTE: split is forced to TEST inside the script."

python -u scripts/eval_checkpoints.py