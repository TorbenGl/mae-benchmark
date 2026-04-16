#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=1024, node007 LOCAL scratch, real data
# Data source: /scratch/imagenet21k_arrow (Arrow format, load_from_disk)
# OOM budget: 26 workers × 4 prefetch × 1024 imgs × 0.6 MB = 64 GB prefetch buffer
# node007: 64 CPUs, 157 GB RAM, H200 NVL 140 GB, local NVMe/SSD scratch
#
# PREREQUISITE: run slurm/build_local_arrow_node007.sh first (one-time, ~4–10 h).
# Compare against throughput-n007-bs1024-real (NFS) to measure storage locality gain.
# =============================================================================
#SBATCH --job-name=throughput-n007local-bs1024-real
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n007local-bs1024-real-%j.out
#SBATCH --error=logs/throughput-n007local-bs1024-real-%j.err

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
    --data_mode real \
    --data_path /scratch/imagenet21k_arrow \
    --prefetch_factor 4 \
    --num_workers 26 \
    --output_dir "$REPO_DIR/outputs/throughput_node007local" \
    --node_label n007local \
    --precision bf16-mixed
