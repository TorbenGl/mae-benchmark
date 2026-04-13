#!/bin/bash
# =============================================================================
# Submit all 12 throughput benchmark presets to SLURM.
#
# PREREQUISITES:
#   1. Undrain node005 if needed:
#        scontrol update nodename=node005 state=resume
#
#   2. Install psutil (if not already in venv):
#        source .venv/bin/activate && pip install psutil
#
#   3. Set environment variables:
#        export DATA_PATH=/datasets/imagenet21k
#        export WANDB_API_KEY=your_key_here
#
#   4. Build dataset cache once (for real-mode runs only):
#        python prebuild_dataset_cache.py --data_path /datasets/imagenet21k
#
# PRESETS: ViT-Base | batch sizes 512/1024/2048 | modes real/synthetic | nodes 005/007
# W&B run names: throughput-{n005,n007}-bs{512,1024,2048}-{real,synthetic}
#
# INTERPRETING RESULTS:
#   If perf/throughput_imgs_per_sec is similar for real and synthetic → data loading is fine.
#   If real < synthetic AND io/io_bound_ratio > 0.3 → storage is the bottleneck.
#   io/disk_read_MB_per_sec shows actual block-device reads (reliable on node007;
#   may under-report on node005 if NFS bypasses block-device counters).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/throughput"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 12 throughput benchmark presets..."
echo "  node005 (remote storage): 3 real + 3 synthetic = 6 jobs"
echo "  node007 (local storage):  3 real + 3 synthetic = 6 jobs"
echo ""

submitted=0
for script in \
    throughput_node005_bs512_real.sh       throughput_node005_bs512_synthetic.sh  \
    throughput_node005_bs1024_real.sh      throughput_node005_bs1024_synthetic.sh \
    throughput_node005_bs2048_real.sh      throughput_node005_bs2048_synthetic.sh \
    throughput_node007_bs512_real.sh       throughput_node007_bs512_synthetic.sh  \
    throughput_node007_bs1024_real.sh      throughput_node007_bs1024_synthetic.sh \
    throughput_node007_bs2048_real.sh      throughput_node007_bs2048_synthetic.sh ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable "$path")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u \$USER"
