#!/bin/bash
# =============================================================================
# MAE SLURM Benchmark — ViT-Large, batch=32, Full H200 (80 GB)
#
# BEFORE SUBMITTING: replace H200_FULL_PARTITION_NAME with your partition.
#   sed -i 's/H200_FULL_PARTITION_NAME/your_partition/' h200_full_vit_l_bs32.sh
# =============================================================================
#SBATCH --job-name=mae-vit_l-bs32-h200full
#SBATCH --partition=H200_FULL_PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mae-vit_l-bs32-h200full-%j.out
#SBATCH --error=logs/mae-vit_l-bs32-h200full-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark.py" \
    --model mae_vit_large_patch16 \
    --batch_size 32 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/h200_full_vit_l_bs32" \
    --gpu_label h200_full \
    --precision 16-mixed \
    --num_workers 8
