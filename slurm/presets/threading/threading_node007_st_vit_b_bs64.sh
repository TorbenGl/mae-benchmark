#!/bin/bash
# =============================================================================
# Threading Benchmark — ViT-Base, batch=64, node007 (local storage)
# Variant: single-thread  |  56 workers × 1 thread = 56 cores
# node007: 64 CPUs, 157 GB RAM, H200 NVL 140 GB
# =============================================================================
#SBATCH --job-name=thr-n007-st-vit_b-bs64
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/thr-n007-st-vit_b-bs64-%j.out
#SBATCH --error=logs/thr-n007-st-vit_b-bs64-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_st.py" \
    --model mae_vit_base_patch16 \
    --batch_size 64 \
    --max_steps 50000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/threading_node007" \
    --gpu_label h200_n007_local_st \
    --precision 16-mixed \
    --num_workers 56 \
    --image_col jpg --label_col cls
