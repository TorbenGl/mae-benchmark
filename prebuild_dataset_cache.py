"""
prebuild_dataset_cache.py — Pre-build the HuggingFace Arrow cache for a local dataset.

Run this ONCE in an interactive session before submitting SLURM benchmark jobs.
On first load of a Parquet-shard dataset, HF datasets builds an Arrow index cache
that can take 10+ minutes for large datasets (e.g. ~10 min for ImageNet-21k at
948k examples).  Once the cache exists subsequent loads are near-instant, so SLURM
jobs won't time out waiting for indexing.

Usage:
    python prebuild_dataset_cache.py
    python prebuild_dataset_cache.py --data_path /datasets/imagenet21k
"""

import argparse
import time

# ── configurable default ──────────────────────────────────────────────────────
DEFAULT_DATA_PATH = "~/imagenet21k"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Pre-build HF Arrow cache for a local dataset.")
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the local HF dataset directory (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()

    from datasets import DatasetDict, load_dataset, load_from_disk

    print(f"Dataset path : {args.data_path}")
    print("Attempting load_from_disk (fast, for .save_to_disk() Arrow layout) ...")
    t0 = time.time()

    try:
        ds = load_from_disk(args.data_path)
        if isinstance(ds, DatasetDict):
            keys = list(ds.keys())
            total = sum(len(ds[k]) for k in keys)
            print(f"load_from_disk OK  —  splits: {keys}, total examples: {total:,}")
        else:
            print(f"load_from_disk OK  —  examples: {len(ds):,}")
    except Exception as e:
        print(f"load_from_disk failed ({e}), falling back to load_dataset (Parquet shards) ...")
        ds = load_dataset(args.data_path, split="all", trust_remote_code=False)
        print(f"load_dataset OK  —  examples: {len(ds):,}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — Arrow cache is ready, SLURM jobs will load instantly.")


if __name__ == "__main__":
    main()
