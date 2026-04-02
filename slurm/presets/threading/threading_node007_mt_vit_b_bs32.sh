#!/bin/bash
# =============================================================================
# Threading Benchmark — ViT-Base, batch=32, node007 (local storage)
# Variant: multi-thread  |  16 workers × 4 threads = 64 cores
# node007: 64 CPUs, 157 GB RAM, H200 NVL 140 GB
# =============================================================================
#SBATCH --job-name=thr-n007-mt-vit_b-bs32
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/thr-n007-mt-vit_b-bs32-%j.out
#SBATCH --error=logs/thr-n007-mt-vit_b-bs32-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_mt.py" \
    --model mae_vit_base_patch16 \
    --batch_size 32 \
    --max_steps 50000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_path "$DATA_PATH" \
    --output_dir "$REPO_DIR/outputs/threading_node007" \
    --gpu_label h200_n007_local_mt \
    --precision 16-mixed \
    --num_workers 16 \
    --image_col jpg --label_col cls
