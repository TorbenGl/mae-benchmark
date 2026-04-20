# MAE-Benchmark

PyTorch implementation of Masked Autoencoders (MAE) — a self-supervised vision learning approach from Meta AI's paper "Masked Autoencoders Are Scalable Vision Learners". Supports pre-training, fine-tuning, and linear probing on Vision Transformer (ViT) backbones.

## Project Scope

Only work within this directory (`mae-benchmark/`). Do not reference or modify files outside this project.

## SLURM Commands

When asked to run any SLURM command (`sbatch`, `scontrol`, `scancel`, `squeue`, etc.), always present the command first and wait for explicit user confirmation before executing it.

## Key Entry Points

- `train_benchmark.py` — **SLURM benchmark** entry point (PyTorch Lightning + W&B); use this for cluster performance testing
- `main_pretrain.py` — Original pre-training script (DDP + TensorBoard)
- `main_finetune.py` — Fine-tune pre-trained models for image classification
- `main_linprobe.py` — Linear probing evaluation (frozen encoder + linear classifier)
- `submitit_*.py` — HPC cluster job submission via Submitit

## Core Modules

- `models_mae.py` — MAE architecture; available models: `mae_vit_small_patch16` (22M), `mae_vit_base_patch16` (86M), `mae_vit_large_patch16` (307M), `mae_vit_huge_patch14`
- `models_vit.py` — ViT models for fine-tuning/linear probing: vit_base_patch16, vit_large_patch16, vit_huge_patch14
- `engine_pretrain.py` — Original pre-training loop
- `engine_finetune.py` — Fine-tuning and evaluation loops
- `util/` — Utilities: distributed training, positional embeddings, LARS optimizer, LR scheduling, datasets, augmentation
- `slurm/` — SBATCH presets for H200 full and H200 MIG benchmark runs

## Tech Stack

- Python 3.9
- PyTorch 2.8.0 + CUDA 12.6
- torchvision 0.23.0
- timm ≥ 0.9.0 (upgraded from original 0.3.2 — required for PyTorch 2.x compatibility)
- lightning ≥ 2.3.0 (PyTorch Lightning — used by `train_benchmark.py`)
- wandb (W&B logging — auth via `WANDB_API_KEY` env var, project: `mae-slurm-benchmark`)
- Distributed training via PyTorch DDP + Submitit (for HPC)
- Mixed precision (AMP), RandAugment, CutMix, Mixup

## Environment

- Development: Windows (no CUDA packages — `nvidia-*-cu12` and `triton` are Linux-only, marked with `; sys_platform == "linux"` in requirements.txt)
- Training / benchmarking runs on Linux with CUDA
- Install: `uv pip install --index-strategy unsafe-best-match -r requirements.txt` — PyTorch wheel index is embedded via `--index-url`; `--index-strategy unsafe-best-match` is required (resolves `torch+cu126` from the PyTorch index) but cannot be set inside requirements.txt
- Dataset: ImageNet-21k (`/datasets/imagenet21k` default); column names `image` / `label` (standard HF format)

## Compatibility Notes (PyTorch 2.8 + timm ≥ 0.9)

The original codebase targeted timm 0.3.2 + PyTorch ~1.x. The following fixes have been applied:

- `util/misc.py` — `from torch._six import inf` → `from math import inf` (torch._six removed in PyTorch 1.9)
- `models_mae.py` — removed `qk_scale=None` from all `Block(...)` calls (removed in timm 0.4+)
- `main_pretrain.py` — removed `assert timm.__version__ == "0.3.2"`; `optim_factory.add_weight_decay` → `param_groups_weight_decay`
- `models_vit.py` — fine-tuning only, not used in benchmark; may need further updates for full timm 1.x compatibility

## Node Inventory (as of 2026-04-20)

| Node | CPUs | RAM | GPU | Scratch | Partition | State |
|---|---|---|---|---|---|---|
| node005 | 64 | ~295 GB (TBC) | H200 NVL 140 GB | unknown — run `scontrol show node node005` | `gpu-node` | IDLE |
| node007 | 64 | ~295 GB | H200 NVL 140 GB | `/scratch` (2 TB, `/dev/sdb1`, local) | `gpu-node` | IDLE |

node007 confirmed via `scontrol`: CPUEfctv=64, RealMemory=302247 MB (~295 GB). Both nodes in partition `gpu-node`. Driver 590.48.01, CUDA 13.1. `/scratch` is local to each node — not mounted on the login node.

The previous "remote vs local storage" distinction referred to `/datasets/imagenet21k` (shared network filesystem). Both nodes also have a large local `/scratch` — copy the dataset there to eliminate network I/O as a variable.

node007 local scratch: `/scratch` — 2 TB dedicated disk (`/dev/sdb1`), nearly empty. Use `--data_path /scratch/imagenet21k` for node007 jobs to avoid network I/O.

The low GPU utilization observed on node005 is suspected to be caused by remote storage — the data rack is not co-located. Use `io/dataloader_wait_ms` and `io/io_bound_ratio` in W&B to confirm.

## Benchmark Architecture

`train_benchmark.py` uses PyTorch Lightning (`MAEBenchmarkModule` + `BenchmarkCallback`) wrapping the existing `MaskedAutoencoderViT`. W&B run names follow `{arch}-bs{batch_size}-{gpu_label}`. SBATCH presets are in `slurm/presets/locality/` (12 presets: ViT-B × 6 batch sizes × 2 nodes). Old presets archived in `slurm/presets/backup/`. See `slurm/USAGE.md` for full cluster operations guide.

Presets use `--max_steps 200 --warmup_epochs 0` for step-based timing probes (dataset size-independent). Use `--fast_dev_run` for a single-batch sanity check without real training.

W&B metrics include `io/dataloader_wait_ms`, `io/io_bound_ratio`, and `gpu/utilization_pct` (via pynvml) for diagnosing data-locality bottlenecks. `system/node` and `system/data_path` are logged to W&B config per run.

## Benchmark Workflow

```bash
# 1. Undrain node005 if needed
scontrol update nodename=node005 state=resume

# 2. Set env vars
export DATA_PATH=/datasets/imagenet21k
export WANDB_API_KEY=your_key

# 3. Smoke test — 1 job per node, single batch each
bash slurm/smoke_test.sh

# 4. Fire all 12 locality presets
bash slurm/submit_all.sh
```

`slurm/smoke_test.sh` submits one ViT-B/bs256 job on node005 and one on node007 using `--fast_dev_run`. Verify both complete before running all 12. See `slurm/USAGE.md` for full details.

## Dataset Cache (run once before benchmarking)

HuggingFace datasets builds an Arrow index cache on first load of a Parquet-shard dataset. For ImageNet-21k (~948k examples) this takes ~10 minutes — long enough to exhaust the smoke test time limit before any training starts. Run this **once** in an interactive session before submitting any SLURM jobs:

```bash
python prebuild_dataset_cache.py
# or with a custom path:
python prebuild_dataset_cache.py --data_path /datasets/imagenet21k
```

The default path is `~/imagenet21k` (set `DEFAULT_DATA_PATH` in the script to change it permanently). After the cache is built, subsequent loads are near-instant and SLURM jobs will not time out waiting for indexing.
