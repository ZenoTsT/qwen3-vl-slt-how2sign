#!/bin/bash
#SBATCH --job-name=qwen3vl_slt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00

# Carico anaconda (di solito già caricato, ma per sicurezza)
module load anaconda3/2023.09-0-none-none
module load cuda/12.6.3
module load cudnn/9.8.0.87-12-none-none-cuda-12.6.3

# Attivo l'env
source activate qwen3vl_env

# Mi sposto nella cartella del progetto
cd /homes/ztesta/qwen3-vl-slt-how2sign

# Creo la cartella logs se non esiste
mkdir -p logs

# LANCIO UN TEST LEGGERO PRIMA (smoke test)
python scripts/smoke_test_qwen3vl.py