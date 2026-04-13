#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=2048, node007 (local storage), real data
# OOM budget: 16 workers × 4 prefetch × 2048 imgs × 0.6 MB = 79 GB prefetch buffer
# node007: 64 CPUs, 157 GB RAM, H200 NVL 140 GB, local storage (co-located)
# =============================================================================
#SBATCH --job-name=throughput-n007-bs2048-real
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n007-bs2048-real-%j.out
#SBATCH --error=logs/throughput-n007-bs2048-real-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 2048 \
    --max_steps 200 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode real \
    --data_path "$DATA_PATH" \
    --prefetch_factor 4 \
    --num_workers 16 \
    --output_dir "$REPO_DIR/outputs/throughput_node007" \
    --node_label n007 \
    --precision bf16-mixed \
    --image_col jpg --label_col cls
