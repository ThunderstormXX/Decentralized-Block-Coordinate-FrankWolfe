from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import pickle
import tarfile

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_cnn import ShallowCNNProblem
from dbcfw_bench.objective_multiclass import MulticlassLogisticProblem

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def make_cifar_problem(config: RunConfig, model: str):
    root = Path(config.data_dir or "data") / "cifar10"
    data_root = _ensure_cifar(root)
    total = config.agents * config.samples_per_agent
    images, labels = _load_train(data_root, total)
    order = np.random.default_rng(config.seed).permutation(len(labels))[:total]
    images = _downsample(images[order].astype(float) / 255.0)
    labels = labels[order].astype(int)
    if model == "linear":
        features = _linear_features(images, config.blocks)
        parts = [part for part in np.array_split(features, config.agents)]
        y_parts = [part for part in np.array_split(labels, config.agents)]
        return MulticlassLogisticProblem(parts, y_parts, config.reg, config.box_radius)
    patches = _patches(images)
    patch_parts = [part for part in np.array_split(patches, config.agents)]
    y_parts = [part for part in np.array_split(labels, config.agents)]
    return ShallowCNNProblem(patch_parts, y_parts, config.hidden_dim, config.reg, config.box_radius)


def _ensure_cifar(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "cifar-10-batches-py"
    if data_root.exists():
        return data_root
    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists():
        urlretrieve(URL, archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root)
    return data_root


def _load_train(data_root: Path, need: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, seen = [], [], 0
    for idx in range(1, 6):
        with (data_root / f"data_batch_{idx}").open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
        xs.append(batch["data"])
        ys.extend(batch["labels"])
        seen += len(batch["labels"])
        if seen >= need:
            break
    images = np.vstack(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.asarray(ys, dtype=int)


def _downsample(images: np.ndarray) -> np.ndarray:
    n = images.shape[0]
    return images.reshape(n, 16, 2, 16, 2, 3).mean(axis=(2, 4))


def _linear_features(images: np.ndarray, blocks: int) -> np.ndarray:
    features = np.hstack([images.reshape(len(images), -1), np.ones((len(images), 1))])
    while (features.shape[1] * 10) % blocks:
        features = np.pad(features, ((0, 0), (0, 1)), mode="constant")
    return features


def _patches(images: np.ndarray) -> np.ndarray:
    parts = []
    for y in range(3):
        for x in range(3):
            parts.append(images[:, y:y + 14, x:x + 14, :].reshape(len(images), -1, 3))
    return np.concatenate(parts, axis=2)
