"""
MAE SLURM Benchmark V2 — train_benchmark_v2.py
===============================================
Extends train_benchmark.py with a configurable --prefetch_factor argument.
Everything else is inherited unchanged from train_benchmark.

New argument:
  --prefetch_factor  DataLoader prefetch_factor (default 4; was hardcoded to 2 in v1)

W&B run name format:
  {arch}-bs{batch_size}-pf{prefetch_factor}-nw{num_workers}-{gpu_label}
"""

from train_benchmark import (
    HFDataModule,
    MAEBenchmarkModule,
    BenchmarkCallback,
    get_args_parser,
    _PYNVML_AVAILABLE,
)

import argparse
import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger


# ---------------------------------------------------------------------------
# V2 DataModule — exposes prefetch_factor as a constructor argument
# ---------------------------------------------------------------------------

class HFDataModuleV2(HFDataModule):
    def __init__(self, *args, prefetch_factor: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            shuffle=(self.trainer.world_size == 1),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser_v2():
    parser = get_args_parser()
    parser.description = "MAE SLURM Benchmark V2"
    parser.add_argument(
        "--prefetch_factor", default=4, type=int,
        help="DataLoader prefetch_factor (default 4; v1 hardcoded 2)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    use_max_steps = args.max_steps > 0
    if use_max_steps and args.warmup_steps == 0 and args.warmup_epochs > 0:
        raise ValueError(
            "--max_steps mode requires --warmup_steps (got --warmup_epochs which has no "
            "fixed meaning when epochs are unlimited). Pass --warmup_steps instead."
        )

    torch.set_float32_matmul_precision("high")

    import lightning as L
    L.seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set. W&B will fall back to offline mode or "
              "use a cached login (run `wandb login` once on this machine).")

    arch_short = (args.model
                  .replace("mae_vit_", "vit_")
                  .replace("_patch16", "")
                  .replace("_patch14", ""))
    run_name = (
        f"{arch_short}-bs{args.batch_size}"
        f"-pf{args.prefetch_factor}"
        f"-nw{args.num_workers}"
        f"-{args.gpu_label}"
    )

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
        data_path=args.data_path,
    )

    datamodule = HFDataModuleV2(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        split=args.hf_split,
        image_col=args.image_col,
        label_col=args.label_col,
        prefetch_factor=args.prefetch_factor,
    )

    trainer = L.Trainer(
        max_steps=args.max_steps if use_max_steps else -1,
        max_epochs=-1 if use_max_steps else args.epochs,
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
    parser = get_args_parser_v2()
    args = parser.parse_args()
    main(args)
