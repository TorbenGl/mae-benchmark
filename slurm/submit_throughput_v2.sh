#!/bin/bash
# Submit all Throughput V2 jobs (mid + high configs only).
# Low configs are excluded — they match the v1 baseline already collected.
#
# Usage:
#   export DATA_PATH=/datasets/imagenet21k
#   bash slurm/submit_throughput_v2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESET_DIR="$SCRIPT_DIR/presets/throughput_v2"

mkdir -p "$SCRIPT_DIR/../logs"

SCRIPTS=(
    throughput_v2_node005_bs512_mid.sh
    throughput_v2_node005_bs512_high.sh
    throughput_v2_node005_bs2048_mid.sh
    throughput_v2_node005_bs2048_high.sh
    throughput_v2_node007_bs512_mid.sh
    throughput_v2_node007_bs512_high.sh
    throughput_v2_node007_bs2048_mid.sh
    throughput_v2_node007_bs2048_high.sh
)

for script in "${SCRIPTS[@]}"; do
    result=$(sbatch "$PRESET_DIR/$script")
    echo "Submitted $script -> $result"
done

echo ""
echo "Done. 8 jobs submitted."
echo "Monitor: squeue -u \$USER"
