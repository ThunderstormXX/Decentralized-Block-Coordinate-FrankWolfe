from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from dbcfw_bench.ot_experiment import (
    OTRunConfig,
    SemiRelaxedOTProblem,
    _run_decentralized_ot,
    _squared_cost,
    row_normalized_plan,
    run_dbcfw_ot,
    run_dfw_ot,
    run_bcfw,
    solve_balanced_ot_lp,
)


@dataclass
class ColorTransferConfig:
    source: str = "rocket"
    target: str = "coffee"
    colors: int = 32
    agents: int = 8
    epochs: int = 80
    batch: int = 1
    relaxation: float = 0.08
    cost_noise: float = 0.0
    stepsize: str = "line_search"
    graph: str = "erdos"
    edge_prob: float = 0.45
    geometric_radius: float = 0.55
    seed: int = 202
    graph_seed: int | None = 1202
    log_every: int = 5
    image_size: int = 240
    sample_pixels: int = 25000
    reference_epochs: int = 1000


@dataclass
class ColorPaletteData:
    source_image: np.ndarray
    target_image: np.ndarray
    source_centers: np.ndarray
    target_centers: np.ndarray
    source_weights: np.ndarray
    target_weights: np.ndarray
    source_labels: np.ndarray
    source_shape: tuple[int, int]


def run_color_transfer_experiment(
    config: ColorTransferConfig,
    out_dir: str | Path,
) -> tuple[pd.DataFrame, list[Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    palette = make_color_palette_data(config)
    problem = make_color_transfer_problem(config, palette)
    balanced_plan = solve_balanced_ot_lp(problem)
    reference_frame, reference_plan = run_bcfw(
        problem,
        _run_config(config, batch=1, epochs=config.reference_epochs, log_every=config.reference_epochs * config.colors),
    )
    reference_frame.to_csv(out / "semi_relaxed_reference_results.csv", index=False)
    transition_epochs = _transition_epochs(config.epochs)
    dfw_frame, dfw_plan, dfw_checkpoints = _run_decentralized_ot(
        problem,
        _run_config(config, batch=config.colors),
        method="dfw",
        reference_plan=reference_plan,
        checkpoint_epochs=transition_epochs,
    )
    dbcfw_frame, dbcfw_plan, dbcfw_checkpoints = _run_decentralized_ot(
        problem,
        _run_config(config, batch=config.batch),
        method="dbcfw",
        reference_plan=reference_plan,
        checkpoint_epochs=transition_epochs,
    )
    frame = pd.concat([dfw_frame, dbcfw_frame], ignore_index=True)
    frame.to_csv(out / "color_transfer_results.csv", index=False)
    _dump_config(config, out / "color_transfer_config.yaml")
    np.savez(
        out / "color_transfer_plans.npz",
        semi_relaxed_reference=reference_plan,
        balanced_lp=balanced_plan,
        dfw=dfw_plan,
        dbcfw=dbcfw_plan,
        source_centers=palette.source_centers,
        target_centers=palette.target_centers,
        source_weights=palette.source_weights,
        target_weights=palette.target_weights,
    )
    image_paths = save_color_transfer_images(out / "images", palette, {
        "source": None,
        "target": None,
        "semi_relaxed_reference": reference_plan,
        "balanced_lp": balanced_plan,
        "dfw": dfw_plan,
        "dbcfw": dbcfw_plan,
    })
    plots = [
        plot_figure9_transition(
            palette,
            dbcfw_checkpoints,
            problem,
            out / "plots" / "figure9_dbcfw_color_transition.png",
        ),
        plot_method_transition_grid(
            palette,
            {"DFW": dfw_checkpoints, "DBCFW": dbcfw_checkpoints},
            problem,
            out / "plots" / "dfw_vs_dbcfw_color_transition.png",
        ),
        plot_color_transfer_comparison(
            palette,
            {
                "Source": None,
                "Target": None,
                "Semi-relaxed ref": reference_plan,
                "Balanced LP": balanced_plan,
                "DFW": dfw_plan,
                "DBCFW": dbcfw_plan,
            },
            problem,
            out / "plots" / "color_transfer_comparison.png",
        ),
        plot_color_transfer_metrics(frame, out / "plots" / "color_transfer_metrics.png"),
        plot_palette_transport(
            palette,
            {
                "Semi-relaxed ref": reference_plan,
                "Balanced LP": balanced_plan,
                "DFW": dfw_plan,
                "DBCFW": dbcfw_plan,
            },
            out / "plots" / "palette_transport_heatmaps.png",
        ),
    ]
    return frame, image_paths + plots


def make_color_palette_data(config: ColorTransferConfig) -> ColorPaletteData:
    rng = np.random.default_rng(config.seed)
    source_image = _load_named_image(config.source, config.image_size)
    target_image = _load_named_image(config.target, config.image_size)
    source_centers, source_weights, source_labels, source_shape = _image_palette(
        source_image, config.colors, config.sample_pixels, rng
    )
    target_centers, target_weights, _, _ = _image_palette(
        target_image, config.colors, config.sample_pixels, rng
    )
    source_order = _palette_order(source_centers)
    target_order = _palette_order(target_centers)
    source_centers, source_weights, source_labels = _reorder_source_palette(
        source_centers, source_weights, source_labels, source_order
    )
    target_centers, target_weights = target_centers[target_order], target_weights[target_order]
    return ColorPaletteData(
        source_image=source_image,
        target_image=target_image,
        source_centers=source_centers,
        target_centers=target_centers,
        source_weights=source_weights,
        target_weights=target_weights,
        source_labels=source_labels,
        source_shape=source_shape,
    )


def make_color_transfer_problem(
    config: ColorTransferConfig,
    palette: ColorPaletteData,
) -> SemiRelaxedOTProblem:
    base_cost = _squared_cost(palette.source_centers, palette.target_centers)
    rng = np.random.default_rng(config.seed + 7001)
    local_costs = np.empty((config.agents, config.colors, config.colors), dtype=float)
    for agent in range(config.agents):
        if config.cost_noise > 0.0:
            noise = config.cost_noise * rng.normal(size=base_cost.shape)
            local_costs[agent] = np.clip(base_cost + noise, 0.0, None)
        else:
            local_costs[agent] = base_cost
    return SemiRelaxedOTProblem(
        source_weights=palette.source_weights,
        target_weights=palette.target_weights,
        source_points=palette.source_centers,
        target_points=palette.target_centers,
        local_costs=local_costs,
        relaxation=config.relaxation,
    )


def transfer_image(palette: ColorPaletteData, plan: np.ndarray) -> np.ndarray:
    row_plan = row_normalized_plan(plan)
    fallback = _nearest_target_colors(palette.source_centers, palette.target_centers)
    mapped_centers = row_plan @ palette.target_centers
    row_mass = plan.sum(axis=1)
    mapped_centers[row_mass <= 1e-14] = fallback[row_mass <= 1e-14]
    pixels = mapped_centers[palette.source_labels].reshape((*palette.source_shape, 3))
    return np.clip(pixels, 0.0, 1.0)


def save_color_transfer_images(
    out_dir: Path,
    palette: ColorPaletteData,
    plans: dict[str, np.ndarray | None],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, plan in plans.items():
        if name == "source":
            image = palette.source_image
        elif name == "target":
            image = palette.target_image
        else:
            if plan is None:
                continue
            image = transfer_image(palette, plan)
        path = out_dir / f"{name}.png"
        Image.fromarray((255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def plot_figure9_transition(
    palette: ColorPaletteData,
    checkpoints: dict[float, np.ndarray],
    problem: SemiRelaxedOTProblem,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted(checkpoints)
    fig, axes = plt.subplots(2, len(epochs) + 2, figsize=(3.1 * (len(epochs) + 2), 6.3))
    axes[0, 0].imshow(palette.source_image)
    axes[0, 0].set_title("source")
    axes[0, 1].imshow(palette.target_image)
    axes[0, 1].set_title("target style")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    vmax = max(float(plan.max()) for plan in checkpoints.values())
    for col, epoch in enumerate(epochs, start=2):
        plan = checkpoints[epoch]
        axes[0, col].imshow(transfer_image(palette, plan))
        axes[0, col].set_title(f"epoch={epoch:g}\ngap={problem.duality_gap(plan):.1e}")
        axes[1, col].imshow(row_normalized_plan(plan), origin="lower", aspect="auto", cmap="magma", vmax=1.0)
        axes[1, col].set_title("row-normalized T")
        axes[1, col].set_xlabel("target color")
        axes[1, col].set_ylabel("source color")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Figure 9 analogue: DBCFW color transfer and row-normalized transport, m=n=32")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_transition_grid(
    palette: ColorPaletteData,
    checkpoints_by_method: dict[str, dict[float, np.ndarray]],
    problem: SemiRelaxedOTProblem,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted(next(iter(checkpoints_by_method.values())))
    methods = list(checkpoints_by_method)
    fig, axes = plt.subplots(len(methods) * 2, len(epochs), figsize=(3.0 * len(epochs), 9.6), squeeze=False)
    for method_index, method in enumerate(methods):
        for col, epoch in enumerate(epochs):
            plan = checkpoints_by_method[method][epoch]
            image_ax = axes[2 * method_index, col]
            heat_ax = axes[2 * method_index + 1, col]
            image_ax.imshow(transfer_image(palette, plan))
            image_ax.set_title(f"{method}, epoch={epoch:g}\ngap={problem.duality_gap(plan):.1e}")
            heat_ax.imshow(row_normalized_plan(plan), origin="lower", aspect="auto", cmap="magma", vmax=1.0)
            heat_ax.set_title(f"{method} row-normalized T")
            for ax in (image_ax, heat_ax):
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("Decentralized DFW vs DBCFW: color-transfer transition")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_color_transfer_comparison(
    palette: ColorPaletteData,
    plans: dict[str, np.ndarray | None],
    problem: SemiRelaxedOTProblem,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(plans), figsize=(3.2 * len(plans), 3.7))
    for ax, (label, plan) in zip(axes, plans.items()):
        if label == "Source":
            image = palette.source_image
            title = "source"
        elif label == "Target":
            image = palette.target_image
            title = "target style"
        else:
            image = transfer_image(palette, plan)  # type: ignore[arg-type]
            title = f"{label}\ngap={problem.duality_gap(plan):.1e}"  # type: ignore[arg-type]
        ax.imshow(image)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Color-transfer outputs")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_color_transfer_metrics(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    panels = [
        ("oracle_epochs", "duality_gap", True, "duality gap vs oracle epochs"),
        ("communication_rounds", "duality_gap", True, "duality gap vs communication rounds"),
        ("oracle_epochs", "objective", False, "objective"),
        ("oracle_epochs", "consensus_error", True, "consensus error"),
    ]
    for ax, (x_col, y_col, log_y, title) in zip(axes.ravel(), panels):
        for method, group in frame.groupby("method", sort=False):
            data = group.sort_values(x_col)
            y = data[y_col].clip(lower=1e-16) if log_y else data[y_col]
            ax.plot(data[x_col], y, marker="o", markersize=2, linewidth=1.4, label=method.upper())
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
    fig.suptitle("Color-transfer optimization diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_palette_transport(
    palette: ColorPaletteData,
    plans: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(plans), figsize=(4.1 * len(plans), 7.2), squeeze=False)
    for col, (label, plan) in enumerate(plans.items()):
        axes[0, col].imshow(plan, origin="lower", aspect="auto", cmap="viridis")
        axes[0, col].set_title(f"{label}: T")
        axes[1, col].imshow(row_normalized_plan(plan), origin="lower", aspect="auto", cmap="magma", vmax=1.0)
        axes[1, col].set_title(f"{label}: row-normalized T")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xlabel("target color")
            ax.set_ylabel("source color")
    fig.suptitle("Transport heatmaps over sorted RGB palettes")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _image_palette(
    image: np.ndarray,
    colors: int,
    sample_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    height, width, _ = image.shape
    pixels = image.reshape(-1, 3)
    sample_size = min(sample_pixels, len(pixels))
    sample_idx = rng.choice(len(pixels), size=sample_size, replace=False)
    sample = pixels[sample_idx]
    centers = _kmeans(sample, colors, rng)
    labels = _assign_nearest(pixels, centers)
    counts = np.bincount(labels, minlength=colors).astype(float)
    weights = counts / counts.sum()
    for idx in range(colors):
        mask = labels == idx
        if np.any(mask):
            centers[idx] = pixels[mask].mean(axis=0)
    return centers, weights, labels, (height, width)


def _kmeans(sample: np.ndarray, clusters: int, rng: np.random.Generator, iterations: int = 30) -> np.ndarray:
    init_idx = rng.choice(len(sample), size=clusters, replace=False)
    centers = sample[init_idx].copy()
    for _ in range(iterations):
        labels = _assign_nearest(sample, centers)
        next_centers = centers.copy()
        for idx in range(clusters):
            mask = labels == idx
            if np.any(mask):
                next_centers[idx] = sample[mask].mean(axis=0)
            else:
                next_centers[idx] = sample[int(rng.integers(0, len(sample)))]
        if np.linalg.norm(next_centers - centers) < 1e-7:
            break
        centers = next_centers
    return np.clip(centers, 0.0, 1.0)


def _assign_nearest(points: np.ndarray, centers: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
    labels = np.empty(len(points), dtype=int)
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        dist = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:start + len(chunk)] = np.argmin(dist, axis=1)
    return labels


def _palette_order(centers: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(centers.reshape(1, -1, 3)).reshape(-1, 3)
    luminance = centers @ np.array([0.2126, 0.7152, 0.0722])
    return np.lexsort((luminance, hsv[:, 2], hsv[:, 1], hsv[:, 0]))


def _reorder_source_palette(
    centers: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return centers[order], weights[order], inverse[labels]


def _nearest_target_colors(source_centers: np.ndarray, target_centers: np.ndarray) -> np.ndarray:
    labels = _assign_nearest(source_centers, target_centers)
    return target_centers[labels]


def _load_named_image(name: str, image_size: int) -> np.ndarray:
    path = Path(name)
    if path.exists():
        image = Image.open(path).convert("RGB")
    else:
        image = Image.fromarray(_load_skimage_sample(name)).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
    canvas.paste(image, offset)
    return np.asarray(canvas, dtype=float) / 255.0


def _load_skimage_sample(name: str) -> np.ndarray:
    try:
        from skimage import data
    except Exception:
        return _fallback_image(name)
    if hasattr(data, name):
        image = getattr(data, name)()
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image
    return _fallback_image(name)


def _fallback_image(name: str) -> np.ndarray:
    seed = sum((idx + 1) * byte for idx, byte in enumerate(name.encode("utf-8"))) % (2**32)
    rng = np.random.default_rng(seed)
    height = width = 320
    x = np.linspace(0.0, 1.0, width)
    y = np.linspace(0.0, 1.0, height)
    xx, yy = np.meshgrid(x, y)
    phase = rng.uniform(0.0, 1.0, size=3)
    image = np.stack([
        0.5 + 0.5 * np.sin(2.0 * np.pi * (xx + phase[0])),
        0.5 + 0.5 * np.sin(2.0 * np.pi * (yy + phase[1])),
        0.5 + 0.5 * np.sin(2.0 * np.pi * (xx + yy + phase[2])),
    ], axis=2)
    return (255.0 * image).astype(np.uint8)


def _run_config(
    config: ColorTransferConfig,
    batch: int,
    epochs: int | None = None,
    log_every: int | None = None,
) -> OTRunConfig:
    return OTRunConfig(
        methods=("dfw", "dbcfw"),
        m=config.colors,
        n=config.colors,
        agents=config.agents,
        epochs=config.epochs if epochs is None else epochs,
        batch=batch,
        relaxation=config.relaxation,
        cost_noise=config.cost_noise,
        stepsize=config.stepsize,
        graph=config.graph,
        edge_prob=config.edge_prob,
        geometric_radius=config.geometric_radius,
        seed=config.seed,
        graph_seed=config.graph_seed,
        log_every=config.log_every if log_every is None else log_every,
    )


def _transition_epochs(epochs: int) -> tuple[float, ...]:
    points = [0.0, 1.0, max(2.0, 0.05 * epochs), max(3.0, 0.2 * epochs), max(4.0, 0.5 * epochs), float(epochs)]
    return tuple(sorted({round(value, 6) for value in points}))


def _dump_config(config: ColorTransferConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=True)
