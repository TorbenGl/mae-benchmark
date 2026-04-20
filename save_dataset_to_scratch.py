"""
save_dataset_to_scratch.py — Convert the Parquet dataset to Arrow format on local /scratch.

Why: HuggingFace Parquet shards require building an index cache on first load (~10 min).
Arrow (save_to_disk) is memory-mapped with O(1) random access — much faster for training.
Storing it on /scratch (local NVMe) eliminates network I/O during benchmarks.

Run ONCE in an interactive session on the target node:
    python save_dataset_to_scratch.py
    python save_dataset_to_scratch.py --src /datasets/imagenet21k --dst /scratch/imagenet21k_arrow

After this completes, point SLURM jobs at the Arrow path:
    export DATA_PATH=/scratch/imagenet21k_arrow

The training scripts (HFDataModule) try load_from_disk() first, which picks up Arrow automatically.
"""

import argparse
import os
import time

DEFAULT_SRC = os.path.expanduser("~/imagenet21k")
DEFAULT_DST = "/scratch/imagenet21k_arrow"


def main():
    parser = argparse.ArgumentParser(description="Convert HF Parquet dataset to Arrow on /scratch")
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help=f"Source dataset path (Parquet shards, default: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help=f"Destination path for Arrow dataset (default: {DEFAULT_DST})")
    parser.add_argument("--num_proc", default=8, type=int,
                        help="Parallel workers for the conversion (default: 8)")
    args = parser.parse_args()

    from datasets import load_dataset, load_from_disk, DatasetDict

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print(f"Workers: {args.num_proc}")

    if os.path.exists(args.dst):
        print(f"\nDestination already exists — loading to verify ...")
        ds = load_from_disk(args.dst)
        if isinstance(ds, DatasetDict):
            total = sum(len(ds[k]) for k in ds)
        else:
            total = len(ds)
        print(f"Already done: {total:,} examples at {args.dst}")
        return

    print("\nLoading source dataset (Parquet) ...")
    t0 = time.time()
    try:
        ds = load_from_disk(args.src)
        print(f"load_from_disk OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"load_from_disk failed ({e}), trying load_dataset ...")
        ds = load_dataset(args.src, split="all", trust_remote_code=False)
        print(f"load_dataset OK ({time.time()-t0:.1f}s)")

    if isinstance(ds, DatasetDict):
        total = sum(len(ds[k]) for k in ds)
    else:
        total = len(ds)
    print(f"Loaded {total:,} examples in {time.time()-t0:.1f}s")

    print(f"\nSaving as Arrow to {args.dst} ...")
    t1 = time.time()
    ds.save_to_disk(args.dst, num_proc=args.num_proc)
    elapsed = time.time() - t1
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nSet DATA_PATH={args.dst} for local-storage benchmark runs.")


if __name__ == "__main__":
    main()
