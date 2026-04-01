#!/bin/bash
# Submit all 14 locality presets (node005 + node007, all batch sizes).


set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/locality"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting all 14 locality presets..."
echo "  node005 (remote storage): 7 jobs"
echo "  node007 (local storage):  7 jobs"
echo ""

submitted=0
for script in "$PRESETS_DIR"/locality_*.sh; do
    job_id=$(sbatch --parsable "$script")
    echo "  Submitted $(basename "$script") -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
