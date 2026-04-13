#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=1024, node007 (local storage), synthetic data
# Synthetic mode: random tensors, no disk I/O — pure GPU throughput baseline.
# node007: 64 CPUs, 157 GB RAM, H200 NVL 140 GB, local storage (co-located)
# =============================================================================
#SBATCH --job-name=throughput-n007-bs1024-synthetic
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n007-bs1024-synthetic-%j.out
#SBATCH --error=logs/throughput-n007-bs1024-synthetic-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 1024 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode synthetic \
    --prefetch_factor 4 \
    --num_workers 8 \
    --output_dir "$REPO_DIR/outputs/throughput_node007" \
    --node_label n007 \
    --precision bf16-mixed
