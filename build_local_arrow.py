"""
build_local_arrow.py — One-time conversion: parquet NFS → Arrow local scratch.
=================================================================================
Reads ImageNet-21k from remote NFS (parquet shards) and saves it in Arrow format
to /scratch/imagenet21k_arrow. After this runs once, load_from_disk() reads the
Arrow files directly — no HF cache ever needed, all I/O stays on local scratch.

Why Arrow instead of just copying parquet?
  Parquet (1.1 TB) + HF Arrow cache (~1.1 TB) = ~2.2 TB → exceeds 2 TB scratch.
  Arrow-only = ~1.1 TB → fits comfortably, and load_from_disk() is cache-free.

Usage:
  # Via SLURM (recommended — runs on node007 with local scratch):
  sbatch slurm/build_local_arrow_node007.sh

  # Interactively on node007:
  source .venv/bin/activate
  python build_local_arrow.py --src /datasets/imagenet21k --dst /scratch/imagenet21k_arrow

  # After completion, verify:
  ls -lh /scratch/imagenet21k_arrow/
  python -c "from datasets import load_from_disk; ds = load_from_disk('/scratch/imagenet21k_arrow'); print(len(ds), 'examples')"
"""

import argparse
import os
import time

DEFAULT_SRC = os.environ.get("DATA_PATH", "/datasets/imagenet21k")
DEFAULT_DST = "/scratch/imagenet21k_arrow"


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF dataset (parquet or Arrow) to Arrow on local scratch."
    )
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help=f"Source dataset path (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help=f"Destination Arrow path on local scratch (default: {DEFAULT_DST})")
    parser.add_argument("--num_proc", default=8, type=int,
                        help="Parallel processes for save_to_disk (default: 8)")
    args = parser.parse_args()

    from datasets import load_dataset, load_from_disk, DatasetDict

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print(f"Workers: {args.num_proc}")
    print()

    t0 = time.time()

    # Try Arrow format first (already save_to_disk layout — fastest path)
    print("Attempting load_from_disk (Arrow format)...")
    try:
        ds = load_from_disk(args.src)
        if isinstance(ds, DatasetDict):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))
        print(f"  load_from_disk OK — {len(ds):,} examples (source is already Arrow)")
    except Exception as e:
        print(f"  load_from_disk failed ({e})")
        print("  Falling back to load_dataset (parquet shards)...")
        ds = load_dataset(args.src, split="all", trust_remote_code=False)
        print(f"  load_dataset OK — {len(ds):,} examples")

    load_elapsed = time.time() - t0
    print(f"\nLoad completed in {load_elapsed/60:.1f} min")
    print(f"Saving to Arrow at {args.dst} (num_proc={args.num_proc})...")
    print("This will take several hours for a 1 TB+ dataset.\n")

    ds.save_to_disk(args.dst, num_proc=args.num_proc)

    total_elapsed = time.time() - t0
    print(f"\nDone in {total_elapsed/3600:.1f}h")
    print(f"Arrow dataset ready at: {args.dst}")
    print()
    print("Next steps:")
    print(f"  1. Verify: python -c \"from datasets import load_from_disk; "
          f"ds = load_from_disk('{args.dst}'); print(len(ds), 'examples')\"")
    print(f"  2. Run benchmarks: bash slurm/submit_throughput_local.sh")


if __name__ == "__main__":
    main()
