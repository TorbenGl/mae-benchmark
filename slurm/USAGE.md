# SLURM Usage Guide — MAE Benchmark

Reference for running locality benchmark jobs on the cluster without interactive assistance.

---

## Node Inventory

| Node | CPUs | RAM | GPU | Storage | State |
|---|---|---|---|---|---|
| node005 | 8 | 16 GB | H200 NVL 140 GB | **remote** (off-rack) | DRAINED* |
| node007 | 16 | 32 GB | H200 NVL 140 GB | local | IDLE |

Partition for both: `gpu-node`
Driver: 590.48.01 | CUDA 13.1

*node005 was auto-drained 2026-03-30 after a batch job failure. Undrain it before submitting jobs there (see below).

---

## Before You Run

### 1. Undrain node005 (if needed)
```bash
# Check current state
scontrol show node node005 | grep State

# Undrain (requires SLURM admin or appropriate permissions)
scontrol update nodename=node005 state=resume

# Verify
scontrol show node node005 | grep State
# Should show: State=IDLE
```

### 2. Build dataset cache (once per cluster, before first job)
HuggingFace Arrow indexing takes ~10 min on first load — long enough to exhaust the smoke test time limit.
```bash
# Run interactively on any node with storage access
srun --partition=gpu-node --nodelist=node007 --pty \
  bash -c "source .venv/bin/activate && python prebuild_dataset_cache.py --data_path /datasets/imagenet21k"
```

### 3. Set environment variables
```bash
export DATA_PATH=/datasets/imagenet21k
export WANDB_API_KEY=your_key_here   # or use 'wandb login' once on the cluster
export VENV_PATH=/path/to/.venv      # optional; defaults to $REPO_DIR/.venv
```

---

## Smoke Test

Submits 1 job per node (node005 + node007), single batch each, to verify the setup:
```bash
bash slurm/smoke_test.sh
```

Expected output: both jobs complete without error. Check logs:
```bash
tail -f slurm/logs/smoke-n005-<job_id>.out
tail -f slurm/logs/smoke-n007-<job_id>.out
```

---

## Full Locality Benchmark

Submits all 12 presets (6 batch sizes × 2 nodes):
```bash
bash slurm/submit_all.sh
```

Or submit individual presets:
```bash
sbatch slurm/presets/locality/locality_node007_vit_b_bs256.sh
sbatch slurm/presets/locality/locality_node005_vit_b_bs256.sh  # requires node005 undrained
```

---

## Monitoring Jobs

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Watch job queue live
watch -n 5 squeue -u $USER

# Follow a job log in real time
tail -f slurm/logs/locality-n007-vit_b-bs256-<job_id>.out

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Node Administration

```bash
# Check node state
scontrol show node node005
scontrol show node node007

# Undrain a node
scontrol update nodename=node005 state=resume

# Quick storage throughput test (confirms data path is accessible)
srun --nodelist=node005 --partition=gpu-node --pty \
  bash -c "ls /datasets/imagenet21k/*.parquet | head -1 | xargs -I{} dd if={} of=/dev/null bs=1M 2>&1"
```

---

## W&B Metrics Guide

All runs log to project `mae-slurm-benchmark`. Key metrics for the locality comparison:

| Metric | What it measures | Healthy threshold |
|---|---|---|
| `gpu/utilization_pct` | SM compute utilization % | > 80% |
| `io/dataloader_wait_ms` | Time GPU waits per batch for data | < 50 ms |
| `io/io_bound_ratio` | Fraction of time starved of data | < 0.1 |
| `perf/throughput_imgs_per_sec` | End-to-end training throughput | higher = better |
| `gpu/memory_allocated_gb` | GPU VRAM in use | — |

**Interpreting results:**
- `io/io_bound_ratio > 0.3` means the GPU spends >30% of its time waiting for data — data loading is the bottleneck.
- If node005 has a much higher `io/dataloader_wait_ms` than node007 at the same batch size, remote storage is causing the GPU starvation.
- Increasing `--num_workers` can help hide latency by prefetching more aggressively, but is limited by available CPUs (node005 has 8, node007 has 16).

**Filter runs in W&B:**
- Tag: `system/node = node005` vs `system/node = node007`
- Group by `gpu_label` (`h200_node005_remote` vs `h200_node007_local`)
