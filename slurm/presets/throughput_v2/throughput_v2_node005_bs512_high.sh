#!/bin/bash
# =============================================================================
# Throughput V2 — ViT-Base, bs=512, node005 (remote storage), HIGH config
# num_workers=28, prefetch_factor=8
# =============================================================================
#SBATCH --job-name=tv2-n005-bs512-high
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tv2-n005-bs512-high-%j.out
#SBATCH --error=logs/tv2-n005-bs512-high-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_v2.py" \
    --model mae_vit_base_patch16 \
    --batch_size 512 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/throughput_v2_node005" \
    --gpu_label h200_node005_remote \
    --precision 16-mixed \
    --num_workers 28 \
    --prefetch_factor 8 \
    --image_col jpg --label_col cls
