from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.data_mnist import _read_images, _read_labels
from dbcfw_bench.objective_mlp import MLPProblem
from dbcfw_bench.objective_torch_cnn import TorchFashionCNNProblem

BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FILES = {"images": "train-images-idx3-ubyte.gz", "labels": "train-labels-idx1-ubyte.gz"}


def make_fashion_mlp_problem(config: RunConfig, non_iid: bool = False) -> MLPProblem:
    data_dir = Path(config.data_dir or "data") / "fashion"
    data_dir.mkdir(parents=True, exist_ok=True)
    images = _read_images(_download(data_dir, FILES["images"]))
    labels = _read_labels(_download(data_dir, FILES["labels"]))
    if non_iid:
        x_parts, y_parts = _non_iid_parts(images, labels, config)
    else:
        x_parts, y_parts = _iid_parts(images, labels, config)
    return MLPProblem(x_parts, y_parts, config.hidden_dim, config.reg, config.box_radius)


def make_fashion_cnn_problem(config: RunConfig, non_iid: bool = False) -> TorchFashionCNNProblem:
    data_dir = Path(config.data_dir or "data") / "fashion"
    data_dir.mkdir(parents=True, exist_ok=True)
    images = _read_images(_download(data_dir, FILES["images"])).reshape(-1, 28, 28)
    images = images.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))[:, None, :, :]
    labels = _read_labels(_download(data_dir, FILES["labels"]))
    if non_iid:
        x_parts, y_parts = _non_iid_parts(images, labels, config)
    else:
        x_parts, y_parts = _iid_parts(images, labels, config)
    return TorchFashionCNNProblem(x_parts, y_parts, config.hidden_dim, config.reg, config.box_radius)


def _download(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{BASE}/{filename}", path)
    return path


def _iid_parts(images: np.ndarray, labels: np.ndarray, config: RunConfig):
    total = config.agents * config.samples_per_agent
    order = np.random.default_rng(config.seed).permutation(len(labels))[:total]
    x_parts = [part for part in np.array_split(images[order], config.agents)]
    y_parts = [part.astype(int) for part in np.array_split(labels[order], config.agents)]
    return x_parts, y_parts


def _non_iid_parts(images: np.ndarray, labels: np.ndarray, config: RunConfig):
    rng = np.random.default_rng(config.seed)
    x_parts, y_parts = [], []
    for agent in range(config.agents):
        allowed = np.array([agent % 10, (agent + 1) % 10])
        pool = np.flatnonzero(np.isin(labels, allowed))
        take = rng.choice(pool, size=config.samples_per_agent, replace=False)
        x_parts.append(images[take])
        y_parts.append(labels[take].astype(int))
    return x_parts, y_parts
