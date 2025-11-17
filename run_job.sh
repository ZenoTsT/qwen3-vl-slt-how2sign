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
    --workdir /$USER \
    -v /mnt/homeGPU/$USER/:/$USER \
    -e HOME=/$USER \
    nvcr.io/nvidia/pytorch:21.02-py3 \
    bash -lc "pip install -r qwen3-vl-slt-how2sign/requirements.txt && \
              cd qwen3-vl-slt-how2sign/scripts && \
              python smoke_test_qwen3vl.py"

stop_rootless_docker.sh