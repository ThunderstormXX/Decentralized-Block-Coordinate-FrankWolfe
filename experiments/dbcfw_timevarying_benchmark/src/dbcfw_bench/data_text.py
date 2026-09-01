from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import re
import zipfile
import zlib

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_logreg import LogisticProblem

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def make_sms_problem(config: RunConfig) -> LogisticProblem:
    labels, texts = _load_sms(Path(config.data_dir or "data") / "sms")
    total = min(config.agents * config.samples_per_agent, len(labels))
    order = np.random.default_rng(config.seed).permutation(len(labels))[:total]
    x = _hash_features([texts[i] for i in order], config.dim)
    y = labels[order]
    x_parts = [part for part in np.array_split(x, config.agents)]
    y_parts = [part.astype(float) for part in np.array_split(y, config.agents)]
    return LogisticProblem(x_parts, y_parts, config.reg, config.box_radius)


def _load_sms(root: Path) -> tuple[np.ndarray, list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    text_path = root / "SMSSpamCollection"
    if not text_path.exists():
        archive = root / "smsspamcollection.zip"
        if not archive.exists():
            urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extract("SMSSpamCollection", root)
    labels, texts = [], []
    with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            label, text = line.rstrip("\n").split("\t", 1)
            labels.append(1 if label == "spam" else -1)
            texts.append(text)
    return np.asarray(labels, dtype=float), texts


def _hash_features(texts: list[str], dim: int) -> np.ndarray:
    if dim < 2:
        raise ValueError("text dim must leave room for a bias coordinate")
    x = np.zeros((len(texts), dim), dtype=float)
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            x[row, zlib.crc32(token.encode("utf-8")) % (dim - 1)] += 1.0
    norms = np.linalg.norm(x[:, :-1], axis=1, keepdims=True)
    x[:, :-1] /= np.maximum(norms, 1.0)
    x[:, -1] = 1.0
    return x
