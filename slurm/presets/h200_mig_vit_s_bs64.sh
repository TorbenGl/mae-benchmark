#!/bin/bash
# =============================================================================
# MAE SLURM Benchmark — ViT-Small, batch=64, H200 MIG (~10 GB slice)
#
# BEFORE SUBMITTING: replace H200_MIG_PARTITION_NAME with your MIG partition.
#   sed -i 's/H200_MIG_PARTITION_NAME/your_partition/' h200_mig_vit_s_bs64.sh
#
# MIG note: if your cluster uses gres instead of partitions for MIG, add:
#   #SBATCH --gres=gpu:1g.10gb:1
# =============================================================================
#SBATCH --job-name=mae-vit_s-bs64-h200mig
#SBATCH --partition=H200_MIG_PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=logs/mae-vit_s-bs64-h200mig-%j.out
#SBATCH --error=logs/mae-vit_s-bs64-h200mig-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark.py" \
    --model mae_vit_small_patch16 \
    --batch_size 64 \
    --max_steps 200 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/h200_mig_vit_s_bs64" \
    --gpu_label h200_mig \
    --precision 16-mixed \
    --num_workers 4
