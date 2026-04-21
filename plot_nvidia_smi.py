"""
plot_nvidia_smi.py — Plot nvidia-smi dmon log files.

Usage:
    python plot_nvidia_smi.py logs/nvidia-smi-439.log
    python plot_nvidia_smi.py logs/nvidia-smi-439.log --out my_plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dmon(filepath):
    """Parse nvidia-smi dmon output into a dict of column -> list of values."""
    headers = None
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # First comment line has column names, second has units — skip units
                if headers is None:
                    headers = line.lstrip("# ").split()
                continue
            parts = line.split()
            if len(parts) != len(headers):
                continue
            rows.append(parts)

    if not headers or not rows:
        sys.exit(f"No data found in {filepath}")

    data = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(float("nan"))  # "-" or other N/A

    return headers, {k: np.array(v) for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="nvidia-smi dmon log file")
    parser.add_argument("--out", default=None, help="Output PNG path (default: logfile.png)")
    args = parser.parse_args()

    logfile = Path(args.logfile)
    out = Path(args.out) if args.out else logfile.with_suffix(".png")

    headers, data = parse_dmon(logfile)
    t = np.arange(len(next(iter(data.values()))))  # time in seconds

    # Pick columns to plot (use whatever is present)
    panels = [
        ("sm",    "SM Utilization (%)", (0, 100)),
        ("mem",   "Memory Utilization (%)", (0, 100)),
        ("pwr",   "Power (W)", None),
        ("gtemp", "GPU Temp (°C)", None),
    ]
    panels = [(col, label, ylim) for col, label, ylim in panels if col in data]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.5 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    fig.suptitle(logfile.name, fontsize=10)

    for ax, (col, label, ylim) in zip(axes, panels):
        ax.plot(t, data[col], linewidth=0.8, color="steelblue")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        mean_val = np.nanmean(data[col])
        ax.axhline(mean_val, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
                   label=f"mean={mean_val:.1f}")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
