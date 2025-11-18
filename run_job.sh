#!/bin/bash
#SBATCH --job-name=prova_qwen_docker
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=slurm-%j.out

module load rootless-docker
start_rootless_docker.sh

docker run --gpus all --rm \
    -v /mnt/homeGPU/$USER/:/$USER \
    -w /$USER/qwen3-vl-slt-how2sign \
    -e HOME=/$USER \
    nvcr.io/nvidia/pytorch:21.02-py3 \
    bash -lc "
      set -e

      echo '>>> Aggiorno pip + typing_extensions'
      pip install --upgrade pip typing_extensions

      echo '>>> Installo requirements del progetto'
      pip install -r requirements.txt

      echo '>>> Setto PYTHONPATH per vedere src/'
      export PYTHONPATH=\$PYTHONPATH:/$USER/qwen3-vl-slt-how2sign

      echo '>>> Lancio smoke test'
      python scripts/smoke_test_qwen3vl.py
    "

stop_rootless_docker.sh