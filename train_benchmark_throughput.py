"""
MAE SLURM Benchmark — train_benchmark_throughput.py
=====================================================
Throughput benchmark comparing real disk I/O vs. synthetic (random tensor) data.
Diagnoses whether data loading speed is causing GPU starvation or loss spikes.

Data modes:
  --data_mode real       Load HF dataset from disk (existing pipeline)
  --data_mode synthetic  Generate random tensors in workers (no disk I/O)

Batch sizes intended for this benchmark: 512, 1024, 2048.

New W&B metrics vs. other benchmark scripts:
  io/disk_read_MB_per_sec  — actual bytes read from block devices (psutil counter diff)
  io/disk_usage_pct        — filesystem fullness % (every 10 steps, real mode only)

W&B:
  Project : mae-slurm-benchmark
  Run name: throughput-{node_label}-bs{batch_size}-{data_mode}
  Auth    : WANDB_API_KEY environment variable (or `wandb login` once on the cluster)

Single-thread mode (OMP/MKL/torch threads = 1, many workers):
  node005 — 40/26/16 workers for bs512/1024/2048 (remote storage; prefetch budget caps)
  node007 — same worker counts (local storage)
"""

import argparse
import io
import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
from pathlib import Path

try:
    import pynvml as _pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import csv
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torchvision.transforms as transforms
from PIL import Image as PILImage
from datasets import load_dataset, load_from_disk, DatasetDict
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import timm.optim.optim_factory as optim_factory
from timm.layers import set_fused_attn

import models_mae


# ---------------------------------------------------------------------------
# File Logger Callback — writes all step metrics to a TSV independently of W&B
# ---------------------------------------------------------------------------

class FileLoggerCallback(L.Callback):
    """Writes every logged metric to a TSV file on disk, independent of W&B.

    Use this to verify W&B is not misreporting. Each row is one training step.
    File is flushed every step so data survives a job cancellation.
    """

    def __init__(self, filepath: str):
        super().__init__()
        self._filepath = filepath
        self._file = None
        self._writer = None
        self._headers_written = False

    def on_train_start(self, trainer, pl_module):
        Path(self._filepath).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, "w", newline="", buffering=1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if isinstance(v, (int, float)) or hasattr(v, "item")}
        metrics["step"] = trainer.global_step
        metrics["wall_time"] = time.time()
        if not metrics:
            return
        if not self._headers_written:
            self._writer = csv.DictWriter(
                self._file, fieldnames=sorted(metrics.keys()), delimiter="\t",
                extrasaction="ignore",
            )
            self._writer.writeheader()
            self._headers_written = True
        self._writer.writerow({k: metrics.get(k, "") for k in self._writer.fieldnames})
        self._file.flush()

    def on_train_end(self, trainer, pl_module):
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Benchmark Callback — throughput, GPU memory, I/O wait, disk metrics
# ---------------------------------------------------------------------------

class BenchmarkCallback(L.Callback):
    """Measures wall-clock throughput, GPU memory, I/O wait, disk read speed, and disk usage.

    Inherits all metrics from the base benchmark callback and adds:
      io/disk_read_MB_per_sec  — system-wide block device reads/sec via psutil counter diff.
                                 Will be near-zero for synthetic mode and NFS mounts.
      io/disk_usage_pct        — filesystem fullness % for the data path.
                                 Only logged every 10 steps in real mode.
    """

    def __init__(self, data_path: str = "", data_mode: str = "real"):
        super().__init__()
        self._batch_end_time = None
        self._last_wait_ms = 0.0
        self._nvml_handle = None
        self._data_path = data_path
        self._data_mode = data_mode
        self._prev_disk_bytes = None
        self._prev_disk_time = None
        if _PYNVML_AVAILABLE:
            try:
                _pynvml.nvmlInit()
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            except Exception:
                pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._batch_end_time is not None:
            self._last_wait_ms = (now - self._batch_end_time) * 1000
        else:
            self._last_wait_ms = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._batch_start
        self._batch_end_time = time.perf_counter()

        bs = pl_module.hparams.batch_size
        throughput = bs / elapsed
        step_ms = elapsed * 1000
        wait_ms = self._last_wait_ms
        total_ms = wait_ms + step_ms
        io_ratio = wait_ms / total_ms if total_ms > 0 else 0.0

        pl_module.log("perf/throughput_imgs_per_sec", throughput,
                      on_step=True, on_epoch=False, rank_zero_only=True)
        pl_module.log("perf/step_time_ms", step_ms,
                      on_step=True, on_epoch=False, rank_zero_only=True)
        pl_module.log("io/dataloader_wait_ms", wait_ms,
                      on_step=True, on_epoch=False, rank_zero_only=True)
        pl_module.log("io/io_bound_ratio", io_ratio,
                      on_step=True, on_epoch=False, rank_zero_only=True)

        # Virtual disk read rate — decoded bytes that would need to come from disk
        # to sustain this step rate. Meaningful for both modes:
        #   synthetic: shows required disk speed if data were real (no I/O noise)
        #   real:      compare against io/disk_read_MB_per_sec to see the gap
        # Formula: batch_size × 3 × H × W × 4 bytes (FP32 decoded tensor) / total_cycle_s
        if total_ms > 0:
            input_size = pl_module.hparams.get("input_size", 224)
            decoded_bytes_per_img = 3 * input_size * input_size * 4
            virtual_disk_mb_s = (bs * decoded_bytes_per_img / 1e6) / (total_ms / 1000)
            pl_module.log("io/virtual_disk_MB_per_sec", virtual_disk_mb_s,
                          on_step=True, on_epoch=False, rank_zero_only=True)

        if torch.cuda.is_available():
            pl_module.log("gpu/memory_allocated_gb",
                          torch.cuda.memory_allocated() / 1e9,
                          on_step=True, on_epoch=False, rank_zero_only=True)
            pl_module.log("gpu/memory_reserved_gb",
                          torch.cuda.memory_reserved() / 1e9,
                          on_step=True, on_epoch=False, rank_zero_only=True)

        if self._nvml_handle is not None:
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                pl_module.log("gpu/utilization_pct", float(util),
                              on_step=True, on_epoch=False, rank_zero_only=True)
            except Exception:
                pass

        # --- disk read throughput (every step, both modes) ---
        # Uses system-wide block-device counters. Note: NFS/network mounts may not
        # appear in these counters — node007 (local storage) gives reliable readings.
        if _PSUTIL_AVAILABLE:
            try:
                counters = _psutil.disk_io_counters()
                now_time = time.perf_counter()
                if counters is not None and self._prev_disk_bytes is not None:
                    delta_bytes = counters.read_bytes - self._prev_disk_bytes
                    delta_t = now_time - self._prev_disk_time
                    if delta_t > 0:
                        disk_mb_s = (delta_bytes / 1e6) / delta_t
                        pl_module.log("io/disk_read_MB_per_sec", disk_mb_s,
                                      on_step=True, on_epoch=False, rank_zero_only=True)
                self._prev_disk_bytes = counters.read_bytes if counters is not None else None
                self._prev_disk_time = now_time
            except Exception:
                pass

        # --- disk usage % (every 10 steps, real mode only) ---
        if (
            _PSUTIL_AVAILABLE
            and self._data_mode == "real"
            and self._data_path
            and batch_idx % 10 == 0
        ):
            try:
                usage = _psutil.disk_usage(self._data_path).percent
                pl_module.log("io/disk_usage_pct", usage,
                              on_step=True, on_epoch=False, rank_zero_only=True)
            except Exception:
                pass

        # --- system-wide CPU utilization (all cores, all processes) ---
        # W&B's built-in cpu_utilization only tracks the main process.
        # This metric captures the full node CPU load including DataLoader workers,
        # so you can see whether workers are actually saturating cores.
        if _PSUTIL_AVAILABLE and batch_idx % 5 == 0:
            try:
                cpu_pct = _psutil.cpu_percent(interval=None)
                pl_module.log("system/cpu_utilization_all_pct", cpu_pct,
                              on_step=True, on_epoch=False, rank_zero_only=True)
            except Exception:
                pass

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.perf_counter() - self._epoch_start
        pl_module.log("perf/epoch_time_s", epoch_time,
                      on_epoch=True, rank_zero_only=True)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class MAEBenchmarkModule(L.LightningModule):
    """Wraps MaskedAutoencoderViT in a LightningModule for throughput benchmarking."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        blr: float,
        min_lr: float,
        weight_decay: float,
        warmup_steps: int,
        warmup_epochs: int,
        epochs: int,
        mask_ratio: float,
        norm_pix_loss: bool,
        data_path: str = "",
        data_mode: str = "real",
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        set_fused_attn(True)  # enable SDPA / FlashAttention-2 in all timm attention layers
        self.model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss)
        if compile_model:
            self.model = torch.compile(self.model)
        self._oom_events = 0

    def on_train_start(self):
        import wandb
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        if wandb.run is not None:
            wandb.run.summary["model_params_M"] = round(n_params, 1)
            wandb.config.update({
                "system/node": os.environ.get("SLURMD_NODENAME", "unknown"),
                "system/data_path": self.hparams.data_path,
                "system/data_mode": self.hparams.data_mode,
            }, allow_val_change=True)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        try:
            loss, _, _ = self.model(images, mask_ratio=self.hparams.mask_ratio)
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            return loss
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._oom_events += 1
            self.log("oom/events", float(self._oom_events),
                     on_step=True, on_epoch=False, rank_zero_only=True)
            return None  # Lightning skips the optimizer step for None returns

    def configure_optimizers(self):
        eff_batch = self.hparams.batch_size * self.trainer.world_size
        lr = self.hparams.blr * eff_batch / 256

        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.hparams.weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

        total_steps = self.trainer.estimated_stepping_batches

        if self.hparams.warmup_steps > 0:
            warmup_steps = self.hparams.warmup_steps
        else:
            steps_per_epoch = max(total_steps // max(self.hparams.epochs, 1), 1)
            warmup_steps = self.hparams.warmup_epochs * steps_per_epoch

        min_lr_ratio = self.hparams.min_lr / lr if lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + cosine * (1.0 - min_lr_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Datasets / DataModules
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    """Returns random (3, input_size, input_size) tensors — no disk I/O at all.

    Used as the baseline to measure pure GPU throughput without any storage bottleneck.
    Each __getitem__ generates a fresh torch.randn tensor so memory stays bounded
    per-worker (no caching). Compared against HFDataModule to isolate disk effects.
    """

    def __init__(self, length: int = 10_000_000, input_size: int = 224):
        self.length = length
        self.input_size = input_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        return torch.randn(3, self.input_size, self.input_size), 0


class SyntheticDataModule(L.LightningDataModule):
    """DataModule backed by SyntheticDataset — pure GPU throughput baseline."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        input_size: int,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.dataset_train = SyntheticDataset(length=10_000_000, input_size=self.input_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            shuffle=False,  # data is already random per __getitem__
        )


class _HFIterableDataset(torch.utils.data.IterableDataset):
    """Iterable HuggingFace dataset wrapper with per-worker shard assignment.

    Each DataLoader worker is assigned a disjoint subset of shards and reads
    them sequentially — turning random seeks into sequential IO. A shuffle
    buffer provides training randomness without disk seeks.
    """

    def __init__(self, hf_dataset, transform, image_col: str, label_col: str,
                 num_workers: int, shuffle_buffer: int = 2000):
        super().__init__()
        num_shards = max(num_workers, 1)
        self._ds = hf_dataset.to_iterable_dataset(num_shards=num_shards)
        self._ds = self._ds.shuffle(buffer_size=shuffle_buffer, seed=42)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self._num_workers = num_workers

    def _decode(self, item):
        img = item[self.image_col]
        if isinstance(img, (bytes, bytearray)):
            img = PILImage.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, PILImage.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
        else:
            img = PILImage.fromarray(img).convert("RGB")
        return self.transform(img), item[self.label_col]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            ds = self._ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        else:
            ds = self._ds
        for item in ds:
            yield self._decode(item)


class HFDataModule(L.LightningDataModule):
    """Loads a locally-saved HuggingFace dataset for MAE pre-training.

    Two loading modes are tried in order:
      1. load_from_disk(data_path)  — Arrow dataset saved with .save_to_disk()
      2. load_dataset(data_path)    — HF-format folder with parquet shards
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
        input_size: int,
        split: str = "train",
        image_col: str = "image",
        label_col: str = "label",
        prefetch_factor: int = 4,
        shuffle_buffer: int = 2000,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.split = split
        self.image_col = image_col
        self.label_col = label_col
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer = shuffle_buffer

    def setup(self, stage=None):
        try:
            ds = load_from_disk(self.data_path)
            if isinstance(ds, DatasetDict):
                from datasets import concatenate_datasets
                ds = concatenate_datasets(list(ds.values()))
        except Exception:
            ds = load_dataset(self.data_path, split="all", trust_remote_code=False)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dataset_train = _HFIterableDataset(
            ds, transform, self.image_col, self.label_col,
            num_workers=self.num_workers,
            shuffle_buffer=self.shuffle_buffer,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            # No shuffle= — IterableDataset handles randomness via shuffle buffer
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("MAE Throughput Benchmark", add_help=True)

    # Model
    parser.add_argument("--model", default="mae_vit_base_patch16", type=str,
                        choices=["mae_vit_small_patch16", "mae_vit_base_patch16",
                                 "mae_vit_large_patch16"],
                        help="MAE model architecture")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (adds ~2 min warmup, then 15-30%% faster)")

    # Training duration
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size per GPU")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Train for exactly this many steps. Set > 0 to activate.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs — used when --max_steps <= 0")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="LR warmup in steps. Takes priority over --warmup_epochs when > 0.")
    parser.add_argument("--warmup_epochs", default=2, type=int,
                        help="LR warmup epochs — used when --warmup_steps == 0")

    # Optimizer
    parser.add_argument("--blr", default=1e-3, type=float,
                        help="Base LR; actual LR = blr * (batch_size * gpus) / 256")
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)

    # Data
    parser.add_argument("--data_mode", default="real", choices=["real", "synthetic"],
                        help="'real' loads HF dataset from disk; "
                             "'synthetic' uses random tensors (no disk I/O, pure GPU baseline)")
    parser.add_argument("--data_path",
                        default=os.environ.get("DATA_PATH", "/datasets/imagenet21k"),
                        type=str,
                        help="Local HF dataset folder (required for real mode; ignored in synthetic)")
    parser.add_argument("--hf_split", default="train", type=str)
    parser.add_argument("--image_col", default="image", type=str,
                        help="Image column name (use 'jpg' for timm/imagenet-w21-wds)")
    parser.add_argument("--label_col", default="label", type=str,
                        help="Label column name (use 'cls' for timm/imagenet-w21-wds)")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="DataLoader workers. Cap carefully: "
                             "workers × prefetch_factor × batch_size × 0.6 MB = RAM usage")
    parser.add_argument("--prefetch_factor", default=4, type=int,
                        help="DataLoader prefetch depth (default 4). "
                             "Higher reduces GPU wait but increases CPU RAM usage.")
    parser.add_argument("--shuffle_buffer", default=2000, type=int,
                        help="Iterable dataset shuffle buffer size per worker (default 2000). "
                             "Higher = better shuffle quality, more RAM.")

    # Output / logging
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--node_label", default="n007", type=str,
                        help="Short node identifier for W&B run name, e.g. n005 or n007")

    # Hardware / precision
    parser.add_argument("--precision", default="bf16-mixed", type=str,
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--seed", default=42, type=int)

    # Quick sanity check
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run 1 train batch for a quick setup sanity check")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    use_max_steps = args.max_steps > 0
    if use_max_steps and args.warmup_steps == 0 and args.warmup_epochs > 0:
        raise ValueError(
            "--max_steps mode requires --warmup_steps (got --warmup_epochs which has no "
            "fixed meaning without a dataset size). Pass --warmup_steps instead."
        )
    if args.data_mode == "real" and not args.data_path:
        raise ValueError("--data_mode real requires --data_path")

    # Guard: cap workers at (available CPUs - 1) to leave headroom for the main process.
    # Over-subscribing is the #1 cause of dataloader stalls on nodes with few CPUs.
    cpu_count = os.cpu_count() or 1
    if args.num_workers >= cpu_count:
        safe_workers = max(1, cpu_count - 1)
        print(f"WARNING: --num_workers {args.num_workers} >= cpu_count {cpu_count}; "
              f"capping at {safe_workers} to avoid CPU starvation")
        args.num_workers = safe_workers

    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set. W&B will fall back to offline mode or "
              "use a cached login (run `wandb login` once on this machine).")

    run_name = f"throughput-{args.node_label}-bs{args.batch_size}-{args.data_mode}"

    wandb_logger = WandbLogger(
        project="mae-slurm-benchmark",
        name=run_name,
        save_dir=args.output_dir,
        log_model=False,
        save_code=False,
    )

    module = MAEBenchmarkModule(
        model_name=args.model,
        batch_size=args.batch_size,
        blr=args.blr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        data_path=args.data_path if args.data_mode == "real" else "",
        data_mode=args.data_mode,
        compile_model=args.compile,
    )

    if args.data_mode == "real":
        datamodule = HFDataModule(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_size=args.input_size,
            split=args.hf_split,
            image_col=args.image_col,
            label_col=args.label_col,
            prefetch_factor=args.prefetch_factor,
            shuffle_buffer=args.shuffle_buffer,
        )
    else:
        datamodule = SyntheticDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_size=args.input_size,
            prefetch_factor=args.prefetch_factor,
        )

    log_file = Path(args.output_dir) / f"{run_name}.tsv"
    print(f"File log : {log_file}")

    trainer = L.Trainer(
        max_steps=args.max_steps if use_max_steps else -1,
        max_epochs=-1 if use_max_steps else args.epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            BenchmarkCallback(
                data_path=args.data_path if args.data_mode == "real" else "",
                data_mode=args.data_mode,
            ),
            FileLoggerCallback(filepath=str(log_file)),
            LearningRateMonitor(logging_interval="step"),
        ],
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
