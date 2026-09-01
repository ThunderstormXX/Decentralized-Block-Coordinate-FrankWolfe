from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    args = _parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.summary_csv).sort_values("batch")
    frame["alpha"] = frame["batch"] / float(args.local_blocks)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "mathtext.fontset": "cm",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.9), dpi=220)
    ax.plot(
        frame["alpha"],
        frame["rel_dir_error_mean"],
        marker="o",
        lw=1.7,
        ms=4.2,
        color="#2f3b4a",
        label="mean",
    )
    ax.plot(
        frame["alpha"],
        frame["rel_dir_error_p90"],
        marker="s",
        lw=1.2,
        ms=3.8,
        color="#7b8794",
        label="p90",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"block fraction $\alpha=B/n_a$")
    ax.set_ylabel(r"$d(\widehat d_B,d_{\mathrm{full}})$")
    ax.grid(True, which="major", color="#d5dae0", lw=0.55, alpha=0.8)
    ax.grid(True, which="minor", axis="x", color="#e7eaee", lw=0.35, alpha=0.75)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(bottom=-0.04)

    fig.tight_layout()
    fig.savefig(args.out_dir / "structural_svm_stochastic_lmo_distance.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "structural_svm_stochastic_lmo_distance.png", bbox_inches="tight")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--local-blocks", type=int, default=893)
    return parser


if __name__ == "__main__":
    main()
