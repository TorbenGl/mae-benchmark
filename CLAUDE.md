# MAE-Benchmark

PyTorch implementation of Masked Autoencoders (MAE) — a self-supervised vision learning approach from Meta AI's paper "Masked Autoencoders Are Scalable Vision Learners". Supports pre-training, fine-tuning, and linear probing on Vision Transformer (ViT) backbones.

## Project Scope

Only work within this directory (`mae-benchmark/`). Do not reference or modify files outside this project.

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

## Benchmark Architecture

`train_benchmark.py` uses PyTorch Lightning (`MAEBenchmarkModule` + `BenchmarkCallback`) wrapping the existing `MaskedAutoencoderViT`. W&B run names follow `{arch}-bs{batch_size}-{gpu_label}`. SBATCH presets are in `slurm/presets/` with partition name placeholders (`H200_FULL_PARTITION_NAME`, `H200_MIG_PARTITION_NAME`) replaced via `bash slurm/submit_all.sh --configure`.

Presets use `--max_steps 200 --warmup_epochs 0` for step-based timing probes (dataset size-independent). Use `--fast_dev_run` for a single-batch sanity check without real training.

## Benchmark Workflow

```bash
# 1. Configure partition names (once)
export H200_FULL_PART=your_partition
export H200_MIG_PART=your_mig_partition
bash slurm/submit_all.sh --configure

# 2. Smoke test — 2 jobs, 1 batch each, 10-min limit
export DATA_PATH=/datasets/imagenet21k
bash slurm/smoke_test.sh

# 3. Fire all 12 presets
bash slurm/submit_all.sh
```

`slurm/smoke_test.sh` submits one ViT-S job per GPU type using `--fast_dev_run`. Verify both complete before running all 12.
