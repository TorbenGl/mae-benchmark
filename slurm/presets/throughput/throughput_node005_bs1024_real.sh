#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=1024, node005 (remote storage), real data
# OOM budget: 26 workers × 4 prefetch × 1024 imgs × 0.6 MB = 64 GB prefetch buffer
# node005: 64 CPUs, 157 GB RAM, H200 NVL 140 GB, remote (off-rack) storage
# NOTE: node005 must be undrained before submitting:
#   scontrol update nodename=node005 state=resume
# NOTE: psutil.disk_io_counters() may not capture NFS reads on node005.
# =============================================================================
#SBATCH --job-name=throughput-n005-bs1024-real
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n005-bs1024-real-%j.out
#SBATCH --error=logs/throughput-n005-bs1024-real-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 1024 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode real \
    --data_path "$DATA_PATH" \
    --prefetch_factor 4 \
    --num_workers 26 \
    --output_dir "$REPO_DIR/outputs/throughput_node005" \
    --node_label n005 \
    --precision bf16-mixed \
    --image_col jpg --label_col cls
