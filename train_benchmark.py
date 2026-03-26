"""
MAE SLURM Benchmark — train_benchmark.py
=========================================
PyTorch Lightning + W&B training script for cluster performance benchmarking.

Benchmark variables:
  --model       mae_vit_small_patch16 | mae_vit_base_patch16 | mae_vit_large_patch16
  --batch_size  per-GPU batch size

W&B:
  Project : mae-slurm-benchmark
  Run name: {arch}-bs{batch_size}-{gpu_label}
  Auth    : WANDB_API_KEY environment variable (or `wandb login` once on the cluster)
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import timm.optim.optim_factory as optim_factory

import models_mae


# ---------------------------------------------------------------------------
# Benchmark Callback — throughput, step time, GPU memory
# ---------------------------------------------------------------------------

class BenchmarkCallback(L.Callback):
    """Measures wall-clock throughput and GPU memory usage per training step.

    Timing spans from on_train_batch_start to on_train_batch_end, which covers
    the full forward → backward → optimizer step (Lightning internals included).
    torch.cuda.synchronize() is called at both boundaries so CUDA work is
    complete before the clock reads.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._batch_start

        bs = pl_module.hparams.batch_size
        throughput = bs / elapsed

        pl_module.log("perf/throughput_imgs_per_sec", throughput,
                      on_step=True, on_epoch=False, rank_zero_only=True)
        pl_module.log("perf/step_time_ms", elapsed * 1000,
                      on_step=True, on_epoch=False, rank_zero_only=True)

        if torch.cuda.is_available():
            pl_module.log("gpu/memory_allocated_gb",
                          torch.cuda.memory_allocated() / 1e9,
                          on_step=True, on_epoch=False, rank_zero_only=True)
            pl_module.log("gpu/memory_reserved_gb",
                          torch.cuda.memory_reserved() / 1e9,
                          on_step=True, on_epoch=False, rank_zero_only=True)

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
    """Wraps MaskedAutoencoderViT in a LightningModule for benchmark training."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        blr: float,
        min_lr: float,
        weight_decay: float,
        warmup_epochs: int,
        epochs: int,
        mask_ratio: float,
        norm_pix_loss: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss)

    def on_train_start(self):
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        if self.logger:
            self.logger.experiment.summary["model_params_M"] = round(n_params, 1)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss, _, _ = self.model(images, mask_ratio=self.hparams.mask_ratio)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Scale LR by effective batch size (per-GPU bs × world size)
        eff_batch = self.hparams.batch_size * self.trainer.world_size
        lr = self.hparams.blr * eff_batch / 256

        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.hparams.weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

        total_steps = self.trainer.estimated_stepping_batches
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
# DataModule
# ---------------------------------------------------------------------------

class ImageNetDataModule(L.LightningDataModule):
    """Loads ImageFolder dataset from data_path/train."""

    def __init__(self, data_path: str, batch_size: int, num_workers: int, input_size: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dataset_train = datasets.ImageFolder(
            os.path.join(self.data_path, "train"), transform=transform
        )

    def train_dataloader(self):
        # Lightning automatically wraps with DistributedSampler when using DDP
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("MAE SLURM Benchmark", add_help=True)

    # Model
    parser.add_argument("--model", default="mae_vit_base_patch16", type=str,
                        choices=["mae_vit_small_patch16", "mae_vit_base_patch16",
                                 "mae_vit_large_patch16"],
                        help="MAE model architecture")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true",
                        help="Use per-patch normalized pixels as targets")

    # Training
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs (10 gives stable benchmark measurements)")
    parser.add_argument("--warmup_epochs", default=2, type=int,
                        help="LR warmup epochs (keep short for benchmarks)")

    # Optimizer
    parser.add_argument("--blr", default=1e-3, type=float,
                        help="Base LR; actual LR = blr * (batch_size * gpus) / 256")
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)

    # Data
    parser.add_argument("--data_path", default=os.environ.get("DATA_PATH", "/datasets/imagenet"),
                        type=str)
    parser.add_argument("--num_workers", default=8, type=int)

    # Output / logging
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--gpu_label", default="h200_full", type=str,
                        help="GPU type label appended to W&B run name (e.g. h200_full, h200_mig)")

    # Hardware / precision
    parser.add_argument("--precision", default="16-mixed", type=str,
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--seed", default=42, type=int)

    # Quick sanity check (Lightning fast_dev_run)
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run 1 train batch for a quick setup sanity check")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    L.seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set. W&B will fall back to offline mode or "
              "use a cached login (run `wandb login` once on this machine).")

    # Build W&B run name: {arch_short}-bs{batch_size}-{gpu_label}
    # e.g. mae_vit_base_patch16 -> vit_b
    arch_short = (args.model
                  .replace("mae_vit_", "vit_")
                  .replace("_patch16", "")
                  .replace("_patch14", ""))
    run_name = f"{arch_short}-bs{args.batch_size}-{args.gpu_label}"

    wandb_logger = WandbLogger(
        project="mae-slurm-benchmark",
        name=run_name,
        save_dir=args.output_dir,
        log_model=False,
    )

    module = MAEBenchmarkModule(
        model_name=args.model,
        batch_size=args.batch_size,
        blr=args.blr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
    )

    datamodule = ImageNetDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            BenchmarkCallback(),
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
