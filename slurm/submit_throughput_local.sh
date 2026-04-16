#!/bin/bash
# =============================================================================
# Submit the 3 local-scratch throughput benchmarks for node007.
#
# PREREQUISITE: build_local_arrow_node007.sh must have completed successfully.
# Verify before submitting:
#   ls -lh /scratch/imagenet21k_arrow/
#   python -c "from datasets import load_from_disk; ds = load_from_disk('/scratch/imagenet21k_arrow'); print(len(ds), 'examples')"
#
# These 3 jobs (real data from /scratch) complement the existing 12 presets:
#   throughput-n007local-bs{512,1024,2048}-real  ← local Arrow, this script
#   throughput-n007-bs{512,1024,2048}-real        ← NFS parquet, submit_throughput.sh
#   throughput-n007-bs{512,1024,2048}-synthetic   ← no disk, submit_throughput.sh
#
# Compare in W&B to quantify NFS vs. local scratch storage locality effect.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/throughput"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 3 local-scratch throughput presets (node007, real data)..."
echo ""

submitted=0
for script in \
    throughput_node007local_bs512_real.sh  \
    throughput_node007local_bs1024_real.sh \
    throughput_node007local_bs2048_real.sh ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable "$path")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u \$USER"
