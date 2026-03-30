#!/bin/bash
#SBATCH --job-name=demo-parallel
#SBATCH --partition=compute-node
#SBATCH --nodes=1               # each task gets its own node
#SBATCH --ntasks=1              # one process per job
#SBATCH --cpus-per-task=4       # 4 CPUs per task
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --array=2          # 6 jobs, max 4 running in parallel (2 queue)
#SBATCH --output=demo/logs/task_%A_%a.out
#SBATCH --error=demo/logs/task_%A_%a.err

echo "=== SLURM task ${SLURM_ARRAY_TASK_ID} / job ${SLURM_JOB_ID} ==="
echo "Running on: $(hostname)"
echo "Started at: $(date)"
which python
python demo/worker.py

echo "Finished at: $(date)"
