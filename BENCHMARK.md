# MAE SLURM Benchmark

End-to-end guide for benchmarking SLURM cluster performance using MAE (Masked Autoencoder) pre-training.
**Benchmark variables:** model size (ViT-S / ViT-B / ViT-L) × per-GPU batch size.

---

## Table of Contents

1. [Compatibility Audit](#1-compatibility-audit)
2. [Architecture Decision](#2-architecture-decision)
3. [W&B Integration](#3-wb-integration)
4. [New Files & Changes](#4-new-files--changes)
5. [Model Presets](#5-model-presets)
6. [SBATCH Presets](#6-sbatch-presets)
7. [Running a Benchmark](#7-running-a-benchmark)
8. [Interpreting W&B Results](#8-interpreting-wb-results)

---

## 1. Compatibility Audit

The original codebase was written for `timm==0.3.2` + PyTorch ~1.x.
Running it against `PyTorch 2.8.0` and a modern `timm` version reveals the following issues:

| File | Issue | Severity | Fix Applied |
|---|---|---|---|
| `util/misc.py:22` | `from torch._six import inf` — `torch._six` removed in PyTorch 1.9 | **CRITICAL** | `from math import inf` |
| `models_mae.py` | `Block(..., qk_scale=None)` — `qk_scale` removed from timm's `Block` in timm 0.4+ | **CRITICAL** | Remove `qk_scale=None` kwarg |
| `main_pretrain.py:27` | `assert timm.__version__ == "0.3.2"` — will always fail with modern timm | **HIGH** | Remove assert |
| `main_pretrain.py:179` | `optim_factory.add_weight_decay()` — renamed to `param_groups_weight_decay()` in timm 0.6+ | **HIGH** | Update call |
| `main_pretrain.py:21` | `from torch.utils.tensorboard import SummaryWriter` — `tensorboard` not in `requirements.txt` | **MEDIUM** | `tensorboard` pulled in as `lightning` transitive dep; benchmark uses W&B instead |
| `engine_pretrain.py:47` | `torch.cuda.amp.autocast()` — deprecated in PyTorch 2.x (functional but warns) | LOW | Left as-is (still works) |
| `util/misc.py:255` | `torch.cuda.amp.GradScaler()` — same deprecation | LOW | Left as-is |
| `requirements.txt` | `timm` not listed at all | **HIGH** | Added `timm>=0.9.0` |
| `models_vit.py` | Used for fine-tuning only, not benchmarked — not touched | N/A | — |

---

## 2. Architecture Decision

**Choice: PyTorch Lightning + W&B** (not raw profiler)

Reasons:
- Lightning's `Trainer` handles DDP/AMP boilerplate cleanly, letting the benchmark focus on metrics
- `WandbLogger` integrates natively — one line to attach
- `Callback.on_train_batch_start/end` with `torch.cuda.synchronize()` gives accurate wall-clock throughput that includes the full forward→backward→optimizer step
- Lightning automatically handles `DistributedSampler` when strategy=`ddp`, so the SBATCH scripts work multi-GPU without code changes
- W&B system monitor automatically tracks GPU utilization % in the background — no extra code needed

**What we do NOT rewrite:**
- `models_mae.py` — the MAE model is kept as-is (after compatibility fixes)
- `util/` — utilities are kept and reused
- `engine_pretrain.py` / `main_pretrain.py` — left intact for non-benchmark use after compatibility patches

---

## 3. W&B Integration

### Authentication
W&B reads `WANDB_API_KEY` automatically from the environment. Set it before submitting jobs:

```bash
export WANDB_API_KEY=your_api_key_here
```

Or save it permanently on the cluster:
```bash
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
```

You can also use `wandb login` interactively once — the token is cached in `~/.netrc`.

### Project & Run Naming
| W&B field | Value |
|---|---|
| **Project** | `mae-slurm-benchmark` |
| **Run name** | `{arch}-bs{batch_size}-{gpu_label}` |
| Example | `vit_b-bs256-h200_full` |

`--gpu_label` is a CLI argument passed by each SBATCH script so runs from different GPU types are visually separated in the W&B dashboard.

### Logged Metrics

| Metric | Description | Interval |
|---|---|---|
| `train/loss` | MAE reconstruction loss (masked patches only) | step + epoch |
| `train/lr` | Current learning rate | step |
| `perf/throughput_imgs_per_sec` | Images processed per second (per GPU) | step |
| `perf/step_time_ms` | Wall-clock time for fwd+bwd+optimizer step | step |
| `gpu/memory_allocated_gb` | Torch-allocated GPU memory | step |
| `gpu/memory_reserved_gb` | CUDA reserved GPU memory (includes caching allocator) | step |
| `system/gpu_util_*` | GPU utilization % (auto-collected by W&B agent) | ~15s |
| `model_params_M` | Model parameter count in millions | once (summary) |

All `perf/*` and `gpu/*` metrics are logged from rank-0 only to avoid duplicates in DDP runs.

---

## 4. New Files & Changes

### New files

| File | Purpose |
|---|---|
| `train_benchmark.py` | Lightning-based benchmark entry point |
| `slurm/presets/h200_full_*.sh` | SBATCH presets for full H200 GPU |
| `slurm/presets/h200_mig_*.sh` | SBATCH presets for H200 MIG instance |
| `slurm/submit_all.sh` | Submit all 12 presets at once |
| `slurm/smoke_test.sh` | Pre-flight check: 2 jobs (1 full H200 + 1 MIG) with `--fast_dev_run` |
| `BENCHMARK.md` | This document |

### Modified files

| File | Changes |
|---|---|
| `util/misc.py` | `torch._six` → `math` |
| `models_mae.py` | Remove `qk_scale=None`, add `mae_vit_small_patch16` |
| `main_pretrain.py` | Remove timm version assert, update `add_weight_decay` → `param_groups_weight_decay` |
| `requirements.txt` | Add `timm>=0.9.0`, `wandb`, `lightning` |

---

## 5. Model Presets

| Arch | Param | embed_dim | depth | heads | Encoder params |
|---|---|---|---|---|---|
| `mae_vit_small_patch16` | ViT-S | 384 | 12 | 6 | ~22 M |
| `mae_vit_base_patch16` | ViT-B | 768 | 12 | 12 | ~86 M |
| `mae_vit_large_patch16` | ViT-L | 1024 | 24 | 16 | ~307 M |

All share the same MAE decoder (512-dim, 8 blocks).

---

## 6. SBATCH Presets

12 preset scripts covering the full benchmark matrix:

```
models × batch_sizes × gpu_types
  3    ×      2      ×     2     = 12 presets
```

### Partition Placeholders

All scripts contain a placeholder for the partition name that **you must replace**:

| Placeholder | Replace with |
|---|---|
| `H200_FULL_PARTITION_NAME` | Your full H200 partition (e.g. `h200`, `gpu-h200`) |
| `H200_MIG_PARTITION_NAME` | Your H200 MIG partition (e.g. `h200-mig`, `mig-1g10gb`) |

```bash
# Quick replacement (bash, from repo root):
sed -i 's/H200_FULL_PARTITION_NAME/YOUR_PARTITION/g' slurm/presets/h200_full_*.sh
sed -i 's/H200_MIG_PARTITION_NAME/YOUR_PARTITION/g' slurm/presets/h200_mig_*.sh
```

### Preset Matrix

#### Full H200 (80 GB)

| Script | Model | BS/GPU | Notes |
|---|---|---|---|
| `h200_full_vit_s_bs64.sh` | ViT-S | 64 | Small model, small batch |
| `h200_full_vit_s_bs512.sh` | ViT-S | 512 | Small model, large batch |
| `h200_full_vit_b_bs64.sh` | ViT-B | 64 | Base model, small batch |
| `h200_full_vit_b_bs256.sh` | ViT-B | 256 | Base model, large batch |
| `h200_full_vit_l_bs32.sh` | ViT-L | 32 | Large model, small batch |
| `h200_full_vit_l_bs128.sh` | ViT-L | 128 | Large model, large batch |

#### H200 MIG (≈10 GB slice, 1g.10gb)

MIG slices have ~10 GB, so batch sizes are reduced accordingly.

| Script | Model | BS/GPU | Notes |
|---|---|---|---|
| `h200_mig_vit_s_bs16.sh` | ViT-S | 16 | Small model, small batch |
| `h200_mig_vit_s_bs64.sh` | ViT-S | 64 | Small model, large batch |
| `h200_mig_vit_b_bs8.sh` | ViT-B | 8 | Base model, small batch |
| `h200_mig_vit_b_bs32.sh` | ViT-B | 32 | Base model, large batch |
| `h200_mig_vit_l_bs4.sh` | ViT-L | 4 | Large model, small batch |
| `h200_mig_vit_l_bs16.sh` | ViT-L | 16 | Large model, large batch |

---

## 7. Running a Benchmark

### Prerequisites

```bash
# 1. Install dependencies (on Linux cluster)
#    PyTorch wheel index is embedded in requirements.txt — no extra flags needed
uv pip install -r requirements.txt

# 2. Set W&B API key
export WANDB_API_KEY=your_key_here

# 3. Configure partition names (replaces placeholders in all preset scripts)
export H200_FULL_PART=h200          # your actual partition
export H200_MIG_PART=h200-mig       # your actual MIG partition
bash slurm/submit_all.sh --configure

# 4. Set your ImageNet-21k data path
#    Scripts default to $DATA_PATH or /datasets/imagenet21k
export DATA_PATH=/datasets/imagenet21k
```

### Smoke test (run before submitting all jobs)

Before firing all 12 presets, verify partition names, data path, and Python environment with a 2-job smoke test. Each job runs `--fast_dev_run` (1 batch only) with a 10-minute time limit:

```bash
bash slurm/smoke_test.sh
```

Check logs in `slurm/logs/smoke-full-<job_id>.out` and `slurm/logs/smoke-mig-<job_id>.out`. Both should exit cleanly before proceeding.

> **Failure modes:** If `sbatch` itself fails (bad partition name, missing `logs/` dir), `submit_all.sh` stops immediately due to `set -euo pipefail` — no jobs are submitted. If a job starts but crashes (bad data path, OOM, import error), only that job fails; others are unaffected.

### Run all presets

```bash
bash slurm/submit_all.sh
```

### Run a single preset

```bash
sbatch slurm/presets/h200_full_vit_b_bs256.sh
```

### Manual run (full control)

```bash
python train_benchmark.py \
  --model mae_vit_base_patch16 \
  --batch_size 256 \
  --max_steps 200 \
  --warmup_epochs 0 \
  --blr 1e-3 \
  --data_path /datasets/imagenet21k \
  --output_dir ./outputs/vit_b_bs256 \
  --gpu_label h200_full \
  --precision 16-mixed \
  --num_workers 8
```

Use `--fast_dev_run` instead of `--max_steps` for a single-batch sanity check:

```bash
python train_benchmark.py \
  --model mae_vit_small_patch16 \
  --batch_size 64 \
  --data_path /datasets/imagenet21k \
  --gpu_label test \
  --fast_dev_run
```

---

## 8. Interpreting W&B Results

### Key comparison metrics

| What to look at | W&B metric | Meaning |
|---|---|---|
| **Throughput** | `perf/throughput_imgs_per_sec` | Higher = better GPU compute |
| **Memory efficiency** | `gpu/memory_allocated_gb` | How much VRAM each config uses |
| **Step latency** | `perf/step_time_ms` | Lower = faster per-step iteration |
| **GPU utilization** | `system/gpu.0.gpu` (auto) | Should be >90% in compute-bound runs |
| **Loss convergence** | `train/loss_epoch` | Sanity check — all runs should converge similarly |

### Comparing H200 full vs MIG
In the W&B dashboard, filter by `gpu_label` to compare identical model+batch combinations across the two GPU types. The primary signal is `perf/throughput_imgs_per_sec`: MIG instances will be slower but allow more concurrent jobs.

### Expected memory usage (rough estimates, fp16)

| Model | Batch | Estimated VRAM |
|---|---|---|
| ViT-S | 64 | ~3 GB |
| ViT-S | 512 | ~15 GB |
| ViT-B | 64 | ~6 GB |
| ViT-B | 256 | ~18 GB |
| ViT-L | 32 | ~12 GB |
| ViT-L | 128 | ~40 GB |
