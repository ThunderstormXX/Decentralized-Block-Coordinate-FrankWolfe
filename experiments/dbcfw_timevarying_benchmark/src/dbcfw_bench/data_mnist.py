from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import gzip
import struct

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_multiclass import MulticlassLogisticProblem

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {"images": "train-images-idx3-ubyte.gz", "labels": "train-labels-idx1-ubyte.gz"}


def make_mnist_problem(config: RunConfig, non_iid: bool = False) -> MulticlassLogisticProblem:
    data_dir = Path(config.data_dir or "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    images = _read_images(_download(data_dir, FILES["images"]))
    labels = _read_labels(_download(data_dir, FILES["labels"]))
    if non_iid:
        return _make_non_iid(images, labels, config)
    total = min(config.agents * config.samples_per_agent, len(labels))
    order = np.random.default_rng(config.seed).permutation(len(labels))[:total]
    features = _prepare_features(images[order], config.blocks)
    x_parts = [part for part in np.array_split(features, config.agents)]
    y_parts = [part.astype(int) for part in np.array_split(labels[order], config.agents)]
    return MulticlassLogisticProblem(x_parts, y_parts, config.reg, config.box_radius)


def _make_non_iid(images: np.ndarray, labels: np.ndarray, config: RunConfig):
    rng = np.random.default_rng(config.seed)
    x_parts, y_parts = [], []
    for agent in range(config.agents):
        allowed = np.array([agent % 10, (agent + 1) % 10])
        pool = np.flatnonzero(np.isin(labels, allowed))
        take = rng.choice(pool, size=config.samples_per_agent, replace=False)
        x_parts.append(_prepare_features(images[take], config.blocks))
        y_parts.append(labels[take].astype(int))
    return MulticlassLogisticProblem(x_parts, y_parts, config.reg, config.box_radius)


def _download(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{BASE}/{filename}", path)
    return path


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(float) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, count = struct.unpack(">II", handle.read(8))
        return np.frombuffer(handle.read(), dtype=np.uint8, count=count).astype(int)


def _prepare_features(images: np.ndarray, blocks: int) -> np.ndarray:
    features = np.hstack([images, np.ones((images.shape[0], 1))])
    while (features.shape[1] * 10) % blocks:
        features = np.pad(features, ((0, 0), (0, 1)), mode="constant")
    return features
