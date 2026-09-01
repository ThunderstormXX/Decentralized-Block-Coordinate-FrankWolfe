from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CENTRAL_COLORS = {"bcfw": "#0072B2", "fw": "#D55E00"}
DECENTRALIZED_COLORS = {
    "dbcfw": "#009E73",
    "dfw": "#CC79A7",
}


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if args.central:
        central = pd.read_csv(args.central)
        paths.extend(plot_central_3d(central, out / "central"))
    if args.decentralized:
        decentralized = pd.read_csv(args.decentralized)
        paths.extend(plot_decentralized_3d(decentralized, out / "decentralized"))
    print("wrote " + ", ".join(str(path) for path in paths))


def plot_central_3d(frame: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [
        _central_metric_3d(
            frame,
            "primal_suboptimality",
            out / "central_primal_suboptimality_3d.png",
            "Primal suboptimality, paper-style OCR validation",
            log_z=True,
        ),
        _central_metric_3d(
            frame,
            "test_error",
            out / "central_test_error_3d.png",
            "OCR test error, paper-style validation",
            log_z=False,
        ),
        _central_metric_3d(
            frame,
            "objective_gap",
            out / "central_objective_gap_3d.png",
            "FW duality gap, paper-style OCR validation",
            log_z=True,
        ),
    ]


def plot_decentralized_3d(frame: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for lambd in sorted(frame["lambda"].dropna().unique(), reverse=True):
        sub = frame[np.isclose(frame["lambda"], lambd)].copy()
        label = _lambda_label(float(lambd))
        slug = _lambda_slug(float(lambd))
        paths.append(_decentralized_metric_3d(
            sub,
            "objective_gap",
            out / f"decentralized_objective_gap_3d_lambda_{slug}.png",
            f"Decentralized OCR objective gap, lambda={label}",
            log_z=True,
        ))
        paths.append(_decentralized_metric_3d(
            sub,
            "test_error",
            out / f"decentralized_test_error_3d_lambda_{slug}.png",
            f"Decentralized OCR test error, lambda={label}",
            log_z=False,
        ))
    return paths


def _central_metric_3d(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    log_z: bool,
) -> Path:
    data = frame[["solver", "lambda", "effective_passes", metric]].dropna().copy()
    data = data[data["effective_passes"] >= 0]
    z_values = _z_values(data[metric], log_z)
    z_floor = _floor(z_values)
    fig, ax = _figure()
    for (solver, lambd), group in data.groupby(["solver", "lambda"], sort=False):
        group = group.sort_values("effective_passes")
        x = group["effective_passes"].to_numpy(dtype=float)
        y = np.full(len(group), np.log10(float(lambd)))
        z = _z_values(group[metric], log_z)
        color = CENTRAL_COLORS.get(str(solver), "#555555")
        ax.plot(x, y, z, color=color, linewidth=2.4, label=f"{solver}, lambda={_lambda_label(float(lambd))}")
        ax.scatter(x[:: max(1, len(x) // 18)], y[:: max(1, len(y) // 18)], z[:: max(1, len(z) // 18)], s=14, color=color, alpha=0.8)
        _ribbon(ax, x, y, z, z_floor, color)
    _format_common(ax, title, "effective passes", "lambda", _metric_label(metric, log_z))
    _lambda_ticks(ax, data["lambda"].unique())
    ax.view_init(elev=25, azim=-58)
    _legend(ax)
    _save(fig, path)
    return path


def _decentralized_metric_3d(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    log_z: bool,
) -> Path:
    data = frame[["method", "solver", "batch", "blocks", "iteration", metric]].dropna().copy()
    z_values = _z_values(data[metric], log_z)
    z_floor = _floor(z_values)
    fig, ax = _figure()
    for (method, batch), group in data.groupby(["method", "batch"], sort=True):
        group = group.sort_values("iteration")
        blocks = float(group["blocks"].iloc[0])
        frac = float(batch) / blocks
        x = group["iteration"].to_numpy(dtype=float)
        y = np.full(len(group), np.log10(frac))
        z = _z_values(group[metric], log_z)
        color = DECENTRALIZED_COLORS.get(str(method), "#555555")
        style = "--" if method == "dfw" else "-"
        width = 2.8 if method == "dfw" else 2.0
        ax.plot(x, y, z, color=color, linestyle=style, linewidth=width, label=_decentralized_label(str(method), int(batch), int(blocks)))
        ax.scatter(x[:: max(1, len(x) // 14)], y[:: max(1, len(y) // 14)], z[:: max(1, len(z) // 14)], s=16, color=color, alpha=0.85)
        _ribbon(ax, x, y, z, z_floor, color, alpha=0.065 if method == "dfw" else 0.11)
    _format_common(ax, title, "iteration", "B / n", _metric_label(metric, log_z))
    _batch_ticks(ax, data)
    ax.view_init(elev=25, azim=-55)
    _legend(ax)
    _save(fig, path)
    return path


def _figure():
    fig = plt.figure(figsize=(11, 7.6), facecolor="#f7f7f4")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fbfbf8")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.97, 0.97, 0.94, 0.92))
        axis.pane.set_edgecolor("#d0d0c8")
    ax.grid(True, alpha=0.28)
    return fig, ax


def _ribbon(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, floor: float, color: str, alpha: float = 0.09) -> None:
    if len(x) < 2:
        return
    top = list(zip(x, y, z))
    bottom = list(zip(x[::-1], y[::-1], np.full_like(z, floor)[::-1]))
    poly = Poly3DCollection([top + bottom], facecolor=color, alpha=alpha, edgecolor="none")
    ax.add_collection3d(poly)


def _format_common(ax, title: str, xlabel: str, ylabel: str, zlabel: str) -> None:
    ax.set_title(title, pad=18, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel(zlabel, labelpad=10)
    ax.tick_params(axis="both", labelsize=9)


def _legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Line2D] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8, frameon=True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _z_values(values: pd.Series, log_z: bool) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    if log_z:
        return np.log10(np.clip(array, 1e-12, None))
    return array


def _floor(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.nanmin(values) - 0.08 * max(np.nanmax(values) - np.nanmin(values), 1e-9))


def _lambda_ticks(ax, lambdas: np.ndarray) -> None:
    values = sorted(float(x) for x in lambdas)
    ax.set_yticks([np.log10(value) for value in values])
    ax.set_yticklabels([_lambda_label(value) for value in values])


def _batch_ticks(ax, frame: pd.DataFrame) -> None:
    pairs = sorted({(int(row.batch), int(row.blocks)) for row in frame.itertuples()})
    ax.set_yticks([np.log10(batch / blocks) for batch, blocks in pairs])
    ax.set_yticklabels([f"{100 * batch / blocks:.1f}%" if batch < blocks else "100%" for batch, blocks in pairs])


def _metric_label(metric: str, log_z: bool) -> str:
    label = metric.replace("_", " ")
    return f"log10({label})" if log_z else label


def _lambda_label(value: float) -> str:
    if abs(value - 1.0 / 6251.0) <= 1e-8:
        return "1/n"
    return f"{value:g}"


def _lambda_slug(value: float) -> str:
    return _lambda_label(value).replace("/", "_").replace(".", "p").replace("-", "m")


def _decentralized_label(method: str, batch: int, blocks: int) -> str:
    prefix = "DFW" if method == "dfw" else "DBCFW"
    return f"{prefix}, B={batch} ({100 * batch / blocks:.1f}%)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dbcfw_bench.paper_ocr_3d")
    parser.add_argument("--central", type=Path, default=None)
    parser.add_argument("--decentralized", type=Path, default=None)
    parser.add_argument("--out", required=True)
    return parser


if __name__ == "__main__":
    main()
