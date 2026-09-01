from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh


@dataclass
class FigureRunConfig:
    resolution: int = 76
    eigen_count: int = 24
    sample_count: int = 360
    seed: int = 2302


@dataclass
class SpectralScene:
    xs: np.ndarray
    ys: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    mask: np.ndarray
    valid_ij: np.ndarray
    points: np.ndarray
    index_map: np.ndarray
    geodesic_graph: object
    evals: np.ndarray
    evecs: np.ndarray

    @property
    def n(self) -> int:
        return int(self.mask.shape[0])

    def nearest_idx(self, point: tuple[float, float]) -> int:
        target = np.array(point, dtype=float)
        return int(np.argmin(np.sum((self.points - target) ** 2, axis=1)))

    def field(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.mask.shape, np.nan, dtype=float)
        out[self.valid_ij[:, 0], self.valid_ij[:, 1]] = values
        return out


def generate_flow_matching_figures(
    out_dir: str | Path,
    config: FigureRunConfig | None = None,
) -> list[Path]:
    cfg = config or FigureRunConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = _build_scene(cfg)
    rng = np.random.default_rng(cfg.seed)
    paths = [
        _plot_premetric_flows(scene, rng, out / "figure1_premetric_flows.png"),
        _plot_distance_contours(scene, out / "figure3_geodesic_spectral_contours.png"),
        _plot_eigen_density(scene, rng, cfg.sample_count, out / "figure4_eigen_density_samples.png"),
        _plot_boundary_trajectories(scene, rng, cfg.sample_count, out / "figure6_boundary_trajectories.png"),
    ]
    with (out / "figure_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, sort_keys=True)
    return paths


def _build_scene(config: FigureRunConfig) -> SpectralScene:
    n = int(config.resolution)
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    x_grid, y_grid = np.meshgrid(xs, ys)
    mask = _largest_component_mask(_domain_mask(x_grid, y_grid))
    geodesic_graph, valid_ij, index_map = _build_graph(mask, weighted=True, xs=xs, ys=ys)
    lap_graph, _, _ = _build_graph(mask, weighted=False, xs=xs, ys=ys)
    degree = np.asarray(lap_graph.sum(axis=1)).ravel()
    laplacian = csgraph.laplacian(lap_graph, normed=False)
    max_modes = max(2, min(int(config.eigen_count) + 1, laplacian.shape[0] - 2))
    evals, evecs = eigsh(laplacian, k=max_modes, which="SM", tol=1e-4)
    order = np.argsort(evals)
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    keep = evals > 1e-8
    evals = evals[keep][: int(config.eigen_count)]
    evecs = evecs[:, keep][:, : int(config.eigen_count)]
    points = np.column_stack((x_grid[valid_ij[:, 0], valid_ij[:, 1]], y_grid[valid_ij[:, 0], valid_ij[:, 1]]))
    # Rescale graph eigenvectors to comparable visual contrast across grid sizes.
    evecs = evecs * np.sqrt(max(1.0, float(degree.mean())))
    return SpectralScene(xs, ys, x_grid, y_grid, mask, valid_ij, points, index_map, geodesic_graph, evals, evecs)


def _domain_mask(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    outer = (x_grid / 0.98) ** 2 + (y_grid / 0.84) ** 2 <= 1.0
    vertical_wall = (np.abs(x_grid + 0.04) < 0.045) & (y_grid > -0.56) & (y_grid < 0.58)
    island = (x_grid + 0.43) ** 2 + (y_grid - 0.05) ** 2 < 0.12**2
    lower_bay = (x_grid - 0.52) ** 2 + (y_grid + 0.42) ** 2 < 0.15**2
    upper_bay = (x_grid - 0.58) ** 2 + (y_grid - 0.48) ** 2 < 0.13**2
    return outer & ~vertical_wall & ~island & ~lower_bay & ~upper_bay


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    graph, valid_ij, _ = _build_graph(mask, weighted=False, xs=np.arange(mask.shape[1]), ys=np.arange(mask.shape[0]))
    _, labels = csgraph.connected_components(graph, directed=False)
    counts = np.bincount(labels)
    keep_label = int(np.argmax(counts))
    out = np.zeros_like(mask, dtype=bool)
    keep = labels == keep_label
    out[valid_ij[keep, 0], valid_ij[keep, 1]] = True
    return out


def _build_graph(mask: np.ndarray, weighted: bool, xs: np.ndarray, ys: np.ndarray):
    valid_ij = np.argwhere(mask)
    index_map = -np.ones(mask.shape, dtype=int)
    index_map[valid_ij[:, 0], valid_ij[:, 1]] = np.arange(len(valid_ij))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for node, (row, col) in enumerate(valid_ij):
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(row + drow), int(col + dcol)
            if rr < 0 or cc < 0 or rr >= mask.shape[0] or cc >= mask.shape[1]:
                continue
            other = int(index_map[rr, cc])
            if other < 0:
                continue
            rows.append(node)
            cols.append(other)
            if weighted:
                data.append(float(np.hypot(xs[cc] - xs[col], ys[rr] - ys[row])))
            else:
                data.append(1.0)
    graph = coo_matrix((data, (rows, cols)), shape=(len(valid_ij), len(valid_ij))).tocsr()
    return graph, valid_ij, index_map


def _plot_premetric_flows(scene: SpectralScene, rng: np.random.Generator, path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    _draw_circle_flow(axes[0], rng)

    ax = axes[1]
    target_idx = scene.nearest_idx((0.62, 0.30))
    distance = _spectral_distance(scene, target_idx, "biharmonic")
    _draw_distance(ax, scene, distance, cmap="viridis")
    starts = _sample_near(scene, (-0.68, -0.46), 13, 0.13, rng)
    for start in starts:
        coords = _path_coordinates(scene, _greedy_path(scene, distance, int(start), target_idx))
        ax.plot(coords[:, 0], coords[:, 1], color="#16bac5", lw=1.7, alpha=0.74)
        ax.scatter(coords[0, 0], coords[0, 1], s=18, color="#1fd7e5", edgecolor="white", linewidth=0.4)
    ax.scatter(
        scene.points[target_idx, 0], scene.points[target_idx, 1],
        s=110, color="#ffd166", edgecolor="#1f1f1f", linewidth=0.8, zorder=5,
    )
    ax.set_title("General manifold: spectral premetric flow")
    ax.text(
        0.02,
        0.03,
        "biharmonic distance\none eigensolve, many targets",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 5},
    )
    _finish_domain_axis(ax)
    fig.suptitle("Figure-style setup: user-specified premetrics define conditional flows", fontsize=16)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_circle_flow(ax, rng: np.random.Generator) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#222222", lw=1.7)
    start_angle = -2.35
    end_angle = 0.62
    arc = np.linspace(start_angle, end_angle, 120)
    for offset in np.linspace(-0.11, 0.11, 9):
        noisy = arc + offset + 0.008 * rng.normal(size=arc.shape)
        radius = 0.92 + 0.04 * np.cos(np.linspace(0, np.pi, len(arc)))
        ax.plot(radius * np.cos(noisy), radius * np.sin(noisy), color="#00a9b7", lw=1.5, alpha=0.72)
    ax.scatter(np.cos(start_angle), np.sin(start_angle), s=120, color="#1fd7e5", edgecolor="#1f1f1f")
    ax.scatter(np.cos(end_angle), np.sin(end_angle), s=120, color="#ffd166", edgecolor="#1f1f1f")
    ax.annotate("", xy=(0.63, 0.72), xytext=(-0.13, -0.95), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#ef476f"})
    ax.set_title("Simple manifold: exact geodesic")
    ax.text(
        0.04,
        0.05,
        "closed-form path\nsimulation-free target",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 5},
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")


def _plot_distance_contours(scene: SpectralScene, path: Path) -> Path:
    _style()
    source_idx = scene.nearest_idx((-0.64, -0.48))
    fields = [
        ("Geodesic", _geodesic_distance(scene, source_idx), "magma"),
        ("Biharmonic", _spectral_distance(scene, source_idx, "biharmonic"), "viridis"),
        ("Diffusion, tau=0.01", _spectral_distance(scene, source_idx, "diffusion", tau=0.01), "viridis"),
        ("Diffusion, tau=0.20", _spectral_distance(scene, source_idx, "diffusion", tau=0.20), "viridis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.6, 4.7), constrained_layout=True)
    image = None
    for ax, (title, values, cmap) in zip(axes, fields):
        image = _draw_distance(ax, scene, values, cmap=cmap)
        ax.scatter(
            scene.points[source_idx, 0], scene.points[source_idx, 1],
            s=70, color="#ffd166", edgecolor="#111111", linewidth=0.7, zorder=5,
        )
        ax.set_title(title)
        _finish_domain_axis(ax)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, label="normalized distance")
    fig.suptitle("Figure 3-style contours: geodesic vs spectral distances", fontsize=15)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_eigen_density(
    scene: SpectralScene,
    rng: np.random.Generator,
    sample_count: int,
    path: Path,
) -> Path:
    _style()
    density = _eigen_target_density(scene)
    samples = rng.choice(len(scene.points), size=int(sample_count), replace=True, p=density)
    sample_field = _sample_density_field(scene, samples)
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.4), constrained_layout=True)
    for col, ax in enumerate(axes[0]):
        mode = min(col, scene.evecs.shape[1] - 1)
        values = scene.evecs[:, mode]
        image = ax.imshow(
            np.ma.masked_invalid(scene.field(values)),
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="coolwarm",
            interpolation="bilinear",
        )
        ax.contour(scene.x_grid, scene.y_grid, scene.mask.astype(float), levels=[0.5], colors="#303030", linewidths=0.8)
        ax.set_title(f"Eigenfunction {mode + 1}")
        _finish_domain_axis(ax)
        fig.colorbar(image, ax=ax, shrink=0.72)

    _draw_density(axes[1, 0], scene, density, "Target density from eigenfunctions")
    _draw_density(axes[1, 1], scene, sample_field, "Learned density proxy")
    _draw_samples(axes[1, 2], scene, samples, "Generated samples")
    target_idx = scene.nearest_idx((0.55, 0.25))
    distance = _spectral_distance(scene, target_idx, "biharmonic")
    starts = _sample_near(scene, (-0.72, -0.42), 12, 0.16, rng)
    _draw_domain(axes[1, 3], scene)
    for start in starts:
        coords = _path_coordinates(scene, _greedy_path(scene, distance, int(start), target_idx))
        axes[1, 3].plot(coords[:, 0], coords[:, 1], color="#ef476f", lw=1.4, alpha=0.7)
    axes[1, 3].scatter(scene.points[target_idx, 0], scene.points[target_idx, 1], s=80, color="#ffd166", edgecolor="#111")
    axes[1, 3].set_title("Biharmonic target paths")
    _finish_domain_axis(axes[1, 3])
    fig.suptitle("Figure 4-style setup: spectral basis, density, and samples", fontsize=15)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_boundary_trajectories(
    scene: SpectralScene,
    rng: np.random.Generator,
    sample_count: int,
    path: Path,
) -> Path:
    _style()
    cases = [
        ((-0.72, -0.48), (0.62, 0.30), "upper target"),
        ((-0.58, 0.52), (0.70, -0.28), "lower target"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    for row, (source, target, name) in enumerate(cases):
        src_idx = _sample_near(scene, source, max(80, sample_count // 3), 0.12, rng)
        tgt_idx = _sample_near(scene, target, max(80, sample_count // 3), 0.12, rng)
        _draw_domain(axes[row, 0], scene)
        axes[row, 0].scatter(scene.points[src_idx, 0], scene.points[src_idx, 1], s=10, color="#1fd7e5", alpha=0.72, label="source")
        axes[row, 0].scatter(scene.points[tgt_idx, 0], scene.points[tgt_idx, 1], s=10, color="#ffd166", alpha=0.72, label="target")
        axes[row, 0].set_title(f"Source and target distributions: {name}")
        axes[row, 0].legend(frameon=True, loc="lower left")
        _finish_domain_axis(axes[row, 0])

        target_center = scene.nearest_idx(target)
        distance = _spectral_distance(scene, target_center, "biharmonic")
        starts = _sample_near(scene, source, 22, 0.12, rng)
        _draw_distance(axes[row, 1], scene, distance, cmap="viridis")
        for start in starts:
            coords = _path_coordinates(scene, _greedy_path(scene, distance, int(start), target_center))
            axes[row, 1].plot(coords[:, 0], coords[:, 1], color="#f7f7f7", lw=1.0, alpha=0.65)
            axes[row, 1].plot(coords[:, 0], coords[:, 1], color="#ef476f", lw=0.7, alpha=0.78)
        axes[row, 1].scatter(scene.points[starts, 0], scene.points[starts, 1], s=16, color="#1fd7e5", edgecolor="white", linewidth=0.3)
        axes[row, 1].scatter(scene.points[target_center, 0], scene.points[target_center, 1], s=95, color="#ffd166", edgecolor="#111")
        axes[row, 1].set_title(f"RCFM-style trajectories: {name}")
        _finish_domain_axis(axes[row, 1])
    fig.suptitle("Figure 6-style setup: non-trivial boundaries and biharmonic trajectories", fontsize=15)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _geodesic_distance(scene: SpectralScene, source_idx: int) -> np.ndarray:
    return csgraph.dijkstra(scene.geodesic_graph, directed=False, indices=int(source_idx))


def _spectral_distance(
    scene: SpectralScene,
    source_idx: int,
    kind: str,
    tau: float = 0.05,
) -> np.ndarray:
    diff = scene.evecs - scene.evecs[int(source_idx)]
    if kind == "biharmonic":
        weights = 1.0 / np.maximum(scene.evals, 1e-9) ** 2
    elif kind == "diffusion":
        weights = np.exp(-2.0 * float(tau) * scene.evals)
    else:
        raise ValueError(f"unknown spectral distance: {kind}")
    return np.sqrt(np.maximum((diff * diff * weights[None, :]).sum(axis=1), 0.0))


def _draw_distance(ax, scene: SpectralScene, values: np.ndarray, cmap: str):
    normalized = _normalize(values)
    image = ax.imshow(
        np.ma.masked_invalid(scene.field(normalized)),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=cmap,
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )
    field = scene.field(normalized)
    ax.contour(scene.x_grid, scene.y_grid, field, levels=np.linspace(0.12, 0.9, 7), colors="white", linewidths=0.45, alpha=0.58)
    ax.contour(scene.x_grid, scene.y_grid, scene.mask.astype(float), levels=[0.5], colors="#242424", linewidths=0.9)
    return image


def _draw_density(ax, scene: SpectralScene, values: np.ndarray, title: str) -> None:
    field = scene.field(_normalize(values))
    ax.imshow(np.ma.masked_invalid(field), origin="lower", extent=(-1, 1, -1, 1), cmap="magma", interpolation="bilinear")
    ax.contour(scene.x_grid, scene.y_grid, scene.mask.astype(float), levels=[0.5], colors="#242424", linewidths=0.9)
    ax.set_title(title)
    _finish_domain_axis(ax)


def _draw_samples(ax, scene: SpectralScene, samples: np.ndarray, title: str) -> None:
    _draw_domain(ax, scene)
    ax.scatter(scene.points[samples, 0], scene.points[samples, 1], s=7, color="#ffd166", alpha=0.58, edgecolor="none")
    ax.set_title(title)
    _finish_domain_axis(ax)


def _draw_domain(ax, scene: SpectralScene) -> None:
    ax.imshow(
        np.ma.masked_where(~scene.mask, scene.mask.astype(float)),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap="Greys",
        alpha=0.18,
        interpolation="nearest",
    )
    ax.contour(scene.x_grid, scene.y_grid, scene.mask.astype(float), levels=[0.5], colors="#242424", linewidths=0.9)


def _finish_domain_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-0.92, 0.92)
    ax.set_xticks([])
    ax.set_yticks([])


def _eigen_target_density(scene: SpectralScene) -> np.ndarray:
    modes = min(6, scene.evecs.shape[1])
    coeff = np.array([1.1, -0.75, 0.55, -0.35, 0.28, -0.22])[:modes]
    score = scene.evecs[:, :modes] @ coeff
    score = score / max(float(score.std()), 1e-9)
    blobs = (
        0.9 * _point_kernel(scene, (0.55, 0.25), 0.18)
        + 0.6 * _point_kernel(scene, (-0.58, 0.46), 0.16)
        + 0.45 * _point_kernel(scene, (0.20, -0.62), 0.14)
    )
    density = np.exp(1.25 * score) + 3.5 * blobs
    density = np.maximum(density, 0.0)
    return density / density.sum()


def _point_kernel(scene: SpectralScene, center: tuple[float, float], scale: float) -> np.ndarray:
    delta = scene.points - np.array(center, dtype=float)
    return np.exp(-0.5 * np.sum(delta * delta, axis=1) / (scale * scale))


def _sample_density_field(scene: SpectralScene, samples: np.ndarray) -> np.ndarray:
    grid = np.zeros(scene.mask.shape, dtype=float)
    ij = scene.valid_ij[samples]
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1.0)
    grid = gaussian_filter(grid, sigma=1.2)
    values = grid[scene.valid_ij[:, 0], scene.valid_ij[:, 1]]
    return values / max(float(values.sum()), 1e-12)


def _sample_near(
    scene: SpectralScene,
    center: tuple[float, float],
    count: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = _point_kernel(scene, center, scale)
    weights = weights / weights.sum()
    return rng.choice(len(scene.points), size=int(count), replace=True, p=weights)


def _greedy_path(
    scene: SpectralScene,
    distance: np.ndarray,
    start_idx: int,
    target_idx: int,
    max_steps: int = 500,
) -> np.ndarray:
    current = int(start_idx)
    path = [current]
    visited = {current}
    for _ in range(max_steps):
        if current == int(target_idx) or distance[current] <= distance[int(target_idx)] + 1e-10:
            break
        candidates = _neighbors(scene, current)
        if not candidates:
            break
        values = distance[np.array(candidates)]
        order = np.argsort(values)
        next_idx = int(candidates[int(order[0])])
        if next_idx in visited and len(order) > 1:
            next_idx = int(candidates[int(order[1])])
        if distance[next_idx] > distance[current] and next_idx in visited:
            break
        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx
    return np.array(path, dtype=int)


def _neighbors(scene: SpectralScene, idx: int) -> list[int]:
    row, col = scene.valid_ij[int(idx)]
    out: list[int] = []
    for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = int(row + drow), int(col + dcol)
        if rr < 0 or cc < 0 or rr >= scene.n or cc >= scene.n:
            continue
        other = int(scene.index_map[rr, cc])
        if other >= 0:
            out.append(other)
    return out


def _path_coordinates(scene: SpectralScene, path: np.ndarray) -> np.ndarray:
    return scene.points[path]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanpercentile(values[finite], 97.5))
    if hi <= lo:
        hi = float(np.nanmax(values[finite]))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
