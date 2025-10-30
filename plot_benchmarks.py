import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


EXPERIMENTS: List[Tuple[str, str]] = [
    ("baseline", "baseline.txt"),
    ("bf16", "bf16.txt"),
    ("bf16 + torch.compile", "bf16_compile.txt"),
    ("bf16 + torch.compile + qkv", "bf16_compile_qkv.txt"),
    ("bf16 + torch.compile + qkv + channels_last", "bf16_compile_qkv_chan.txt"),
    ("bf16 + torch.compile + qkv + channels_last + FA3", "bf16_compile_qkv_chan_fa3.txt"),
    (
        "bf16 + torch.compile + qkv + channels_last + FA3 + quant",
        "bf16_compile_qkv_chan_fa3_quant.txt",
    ),
    (
        "bf16 + torch.compile + qkv + channels_last + FA3 + quant + flags",
        "bf16_compile_qkv_chan_fa3_quant_flags.txt",
    ),
    ("fully_optimized", "fully_optimized.txt"),
]

MEAN_PATTERN = re.compile(r"time mean/var:\s*\[.*?\]\s*([0-9.eE+-]+)\s+[0-9.eE+-]+")


def parse_mean_runtime(file_path: str) -> float:
    """Extract the mean runtime (in seconds) from the benchmark log."""
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            match = MEAN_PATTERN.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Unable to find mean runtime in {file_path}")


def load_runtimes(root_dir: str) -> Dict[str, float]:
    runtimes: Dict[str, float] = {}
    for label, filename in EXPERIMENTS:
        path = os.path.join(root_dir, filename)
        runtimes[label] = parse_mean_runtime(path)
    return runtimes


def plot_runtimes(runtimes: Dict[str, float], title: str) -> None:
    labels = list(runtimes.keys())
    means = [runtimes[label] * 1_000 for label in labels]  # convert to milliseconds

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(labels)))
    bars = ax.bar(x_positions, means, color="#4C72B0")
    ax.set_ylabel("Mean Runtime (ms)")
    ax.set_xlabel("Experiment")
    title = title + "_Benchmark_Runtime_Comparison"
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"{value:.3f} ms",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = title + ".png"
    fig.savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dummy benchmark runtimes from experiments_dummy.sh artifacts."
    )
    parser.add_argument(
        "--logs-root",
        default=".",
        help="Directory containing the *.txt benchmark outputs (default: current directory).",
    )
    parser.add_argument(
        "--title",
        default="Dummy",
        help="Title for the generated plot.",
    )
    args = parser.parse_args()

    runtimes = load_runtimes(args.logs_root)
    plot_runtimes(runtimes, args.title)


if __name__ == "__main__":
    main()
