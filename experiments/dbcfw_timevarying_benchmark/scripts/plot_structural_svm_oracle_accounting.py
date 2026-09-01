from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


def main() -> None:
    args = _parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(10.6, 3.6), dpi=220)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.28, 1.28], wspace=0.18)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _draw_graph_panel(axes[0])
    _draw_blocks_panel(
        axes[1],
        title="DFW: all local blocks",
        subtitle=r"$B=n_a=893$",
        mode="full",
    )
    _draw_blocks_panel(
        axes[2],
        title="DBCFW: one sampled block",
        subtitle=r"$B=1$ per agent",
        mode="sampled",
    )

    fig.savefig(args.out_dir / "structural_svm_oracle_accounting_diagram.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "structural_svm_oracle_accounting_diagram.png", bbox_inches="tight")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _panel(ax: plt.Axes) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (0.025, 0.08),
        0.95,
        0.84,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor="#8d98a7",
        facecolor="#fbfbfb",
    )
    ax.add_patch(patch)
    return patch


def _draw_graph_panel(ax: plt.Axes) -> None:
    _panel(ax)
    ax.text(0.08, 0.22, r"$N=7$ agents", fontsize=10.5, color="#4f5b6c", va="center")
    ax.text(0.08, 0.14, r"$n_a=893$ blocks/agent", fontsize=10.5, color="#4f5b6c", va="center")

    points = np.array(
        [
            [0.77, 0.66],
            [0.63, 0.79],
            [0.39, 0.80],
            [0.22, 0.66],
            [0.27, 0.50],
            [0.49, 0.41],
            [0.71, 0.48],
        ]
    )
    edges = [(0, 1), (0, 3), (0, 6), (1, 2), (1, 5), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    for a, b in edges:
        ax.plot(
            [points[a, 0], points[b, 0]],
            [points[a, 1], points[b, 1]],
            color="#c5ccd5",
            lw=1.25,
            zorder=1,
        )
    for idx, (x, y) in enumerate(points, start=1):
        node = plt.Circle((x, y), 0.052, facecolor="white", edgecolor="#4b6f9f", lw=1.25, zorder=2)
        ax.add_patch(node)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=9.5, color="#334155", zorder=3)


def _draw_blocks_panel(ax: plt.Axes, title: str, subtitle: str, mode: str) -> None:
    _panel(ax)
    ax.text(0.07, 0.83, title, fontsize=12, fontweight="normal", va="center")
    ax.text(0.07, 0.75, subtitle, fontsize=10.5, color="#4f5b6c", va="center")

    left = 0.17
    right = 0.92
    row_top = 0.63
    row_gap = 0.07
    row_h = 0.026
    cols = 28
    col_gap = 0.004
    cell_w = (right - left - col_gap * (cols - 1)) / cols

    ax.text(left, 0.69, "local word blocks by agent", fontsize=10.5, va="center", color="#2b3038")
    sampled_cols = [18, 4, 23, 9, 15, 26, 11]

    for row in range(7):
        y = row_top - row * row_gap
        ax.text(0.095, y + row_h / 2, str(row + 1), ha="center", va="center", fontsize=9.5, color="#4f5b6c")
        for col in range(cols):
            x = left + col * (cell_w + col_gap)
            if mode == "full":
                face = "#6f7782"
                edge = "#6f7782"
            elif col == sampled_cols[row]:
                face = "#3f4752"
                edge = "#3f4752"
            else:
                face = "#eef1f4"
                edge = "#d4dae1"
            rect = Rectangle((x, y), cell_w, row_h, facecolor=face, edgecolor=edge, linewidth=0.35)
            ax.add_patch(rect)


if __name__ == "__main__":
    main()
