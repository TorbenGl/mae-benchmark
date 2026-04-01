#!/bin/bash
# =============================================================================
# Data Locality Benchmark — ViT-Base, batch=1024, node005 (remote storage)
# node005: 8 CPUs, 16 GB RAM, H200 NVL 140 GB
# num_workers reduced to 4 (vs 7 for smaller batches) to keep DataLoader prefetch
# buffer (~4 GB at bs=1024) within the 15G RAM limit.
# NOTE: node005 must be undrained before submitting:
#   scontrol update nodename=node005 state=resume
# =============================================================================
#SBATCH --job-name=locality-n005-vit_b-bs1024
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=logs/locality-n005-vit_b-bs1024-%j.out
#SBATCH --error=logs/locality-n005-vit_b-bs1024-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark.py" \
    --model mae_vit_base_patch16 \
    --batch_size 1024 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/locality_node005" \
    --gpu_label h200_node005_remote \
    --precision 16-mixed \
    --num_workers 14 \
    --image_col jpg --label_col cls
