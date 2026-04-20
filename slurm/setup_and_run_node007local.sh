#!/bin/bash
# =============================================================================
# One-shot: build local Arrow dataset on node007 /scratch, then run benchmarks.
#
# Step 1 — build-local-arrow-n007 (no GPU, ~4–10 h)
#   Reads /datasets/imagenet21k from NFS, writes Arrow to /scratch/imagenet21k_arrow
#
# Step 2 — 3 throughput benchmarks (afterok dependency on step 1)
#   throughput-n007local-bs{512,1024,2048}-real
#   Start only after step 1 succeeds.
#
# Usage:
#   bash slurm/setup_and_run_node007local.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/build-local-arrow-n007-<id>.out
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/throughput"

mkdir -p "$SCRIPT_DIR/logs"

# --- Step 1: submit build job ---
echo "Step 1: Submitting Arrow build job (node007, no GPU, ~4-10 h)..."
build_job_id=$(sbatch --parsable "$SCRIPT_DIR/build_local_arrow_node007.sh")
echo "  Build job -> $build_job_id"
echo ""

# --- Step 2: submit benchmarks with afterok dependency ---
echo "Step 2: Submitting 3 throughput benchmarks (will start after job $build_job_id completes)..."
echo ""

submitted=0
for script in \
    throughput_node007local_bs512_real.sh  \
    throughput_node007local_bs1024_real.sh \
    throughput_node007local_bs2048_real.sh ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable --dependency=afterok:$build_job_id "$path")
    echo "  Submitted $script -> job $job_id (depends on $build_job_id)"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. 1 build job + $submitted benchmark jobs submitted."
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Watch build progress:"
echo "  tail -f logs/build-local-arrow-n007-${build_job_id}.out"
echo ""
echo "Verify Arrow dataset after build:"
echo "  python -c \"from datasets import load_from_disk; ds = load_from_disk('/scratch/imagenet21k_arrow'); print(len(ds), 'examples')\""
