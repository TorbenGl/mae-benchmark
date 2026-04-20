#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=1024, node007 (local /scratch Arrow), real data
# node007: 64 CPUs, 302 GB RAM, H200 NVL 140 GB
# RAM budget: 32 workers × 6 prefetch × 1024 imgs × 0.6 MB ≈ 118 GB
# DATA_PATH must point to Arrow dataset on /scratch (run submit_node007_local_pipeline.sh)
# =============================================================================
#SBATCH --job-name=throughput-n007local-bs1024-real
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n007local-bs1024-real-%j.out
#SBATCH --error=logs/throughput-n007local-bs1024-real-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/scratch/imagenet21k_arrow}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "DATA_PATH=$DATA_PATH"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 1024 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode real \
    --data_path "$DATA_PATH" \
    --prefetch_factor 6 \
    --num_workers 32 \
    --output_dir "$REPO_DIR/outputs/throughput_node007" \
    --node_label n007local \
    --precision bf16-mixed \
    --image_col jpg --label_col cls
