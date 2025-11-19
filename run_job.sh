#!/bin/bash
#SBATCH --job-name=qwen3vl_slt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

# Carico anaconda (di solito già caricato, ma per sicurezza)
module load anaconda3/2023.09-0-none-none

# Attivo l'env
source activate qwen3vl_env

# Mi sposto nella cartella del progetto
cd /homes/ztesta/qwen3-vl-slt-how2sign

# Rendo la root del progetto visibile a Python
export PYTHONPATH=/homes/ztesta/qwen3-vl-slt-how2sign:$PYTHONPATH

# Creo la cartella logs se non esiste
mkdir -p logs

# LANCIO UN TEST LEGGERO PRIMA (smoke test)
python scripts/smoke_test_qwen3vl_multiple_frames.py