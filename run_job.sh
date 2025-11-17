#!/bin/bash
#SBATCH --job-name=prova_dios          # job name
#SBATCH --partition=dios               # my queue
#SBATCH --gres=gpu:1                   # num of GPUs
#SBATCH --cpus-per-task=4              # num of CPUs
#SBATCH --mem=20G                      # RAM
#SBATCH --output=slurm-%j.out          # log file

# Attiva conda
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /home/ztesta/ztesta/conda_env_zeno

# Vai nella repo
cd /home/ztesta/ztesta/qwen3-vl-slt-how2sign/scripts

# Lancia il tuo script
python smoke_test_qwen3vl.py