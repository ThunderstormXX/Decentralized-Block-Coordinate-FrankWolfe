from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_logreg import LogisticProblem

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"


def make_mushroom_problem(config: RunConfig) -> LogisticProblem:
    path = _dataset_path(config)
    frame = pd.read_csv(path, header=None, names=["label"] + [f"a{i}" for i in range(22)])
    y = np.where(frame["label"].to_numpy() == "p", 1.0, -1.0)
    x_frame = pd.get_dummies(frame.drop(columns=["label"]), dtype=float)
    features = x_frame.to_numpy(dtype=float)
    features = _pad_columns(features, config.blocks)
    order = np.random.default_rng(config.seed).permutation(len(y))
    x_parts = [part for part in np.array_split(features[order], config.agents)]
    y_parts = [part for part in np.array_split(y[order], config.agents)]
    return LogisticProblem(x_parts, y_parts, config.reg, config.box_radius)


def _dataset_path(config: RunConfig) -> Path:
    data_dir = Path(config.data_dir or "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "agaricus-lepiota.data"
    if not path.exists():
        urlretrieve(URL, path)
    return path


def _pad_columns(features: np.ndarray, blocks: int) -> np.ndarray:
    pad = (-features.shape[1]) % blocks
    if pad == 0:
        return features
    return np.pad(features, ((0, 0), (0, pad)), mode="constant")
