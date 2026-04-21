#!/bin/bash
# Submit the 3 node007 local-scratch throughput benchmarks.
# Requires Arrow dataset already built at /scratch/imagenet21k_arrow.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/throughput"

mkdir -p "$SCRIPT_DIR/logs"

for s in \
    throughput_node007local_bs512_real.sh  \
    throughput_node007local_bs1024_real.sh \
    throughput_node007local_bs2048_real.sh ; do

    job_id=$(sbatch --parsable "$PRESETS_DIR/$s")
    echo "Submitted $s -> job $job_id"
done
