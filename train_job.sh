#!/bin/bash
#SBATCH --job-name=qwen3vl_slt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=tesi_ztesta
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00

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


# ------------------------------
# Lancio training (DDP single-GPU)
# ------------------------------
MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/training/train_qwen3vl.py