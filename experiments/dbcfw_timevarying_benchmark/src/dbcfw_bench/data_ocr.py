from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import gzip
import shutil

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.objective_structural_svm import StructuralSequenceSVMProblem

OCR_URL = "http://ai.stanford.edu/~btaskar/ocr/letter.data.gz"


def make_ocr_structural_svm_problem(config: RunConfig) -> StructuralSequenceSVMProblem:
    train_x, train_y, test_x, test_y = load_taskar_ocr(Path(config.data_dir or "data"), "ocr2")
    total = config.agents * config.blocks
    if total <= len(train_x):
        train_x = train_x[:total]
        train_y = train_y[:total]
    elif config.agents > 1:
        raise ValueError(
            f"requested {total} OCR train examples, but only {len(train_x)} are available"
        )
    x_parts = [list(part) for part in np.array_split(np.asarray(train_x, dtype=object), config.agents)]
    y_parts = [list(part) for part in np.array_split(np.asarray(train_y, dtype=object), config.agents)]
    return StructuralSequenceSVMProblem(
        x_parts,
        y_parts,
        config.reg,
        classes=26,
        position_bias=True,
        test_x=test_x,
        test_y=test_y,
    )


def load_taskar_ocr(
    data_dir: str | Path,
    split: str = "ocr2",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    root = Path(data_dir) / "ocr"
    path = _ensure_letter_data(root)
    words = _read_words(path)
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test_x: list[np.ndarray] = []
    test_y: list[np.ndarray] = []
    for fold, x_seq, y_seq in words:
        train = (split == "ocr2" and fold != 0) or (split == "ocr" and fold == 0)
        if train:
            train_x.append(x_seq)
            train_y.append(y_seq)
        else:
            test_x.append(x_seq)
            test_y.append(y_seq)
    return train_x, train_y, test_x, test_y


def _ensure_letter_data(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "letter.data"
    if path.exists():
        return path
    archive = root / "letter.data.gz"
    if not archive.exists():
        urlretrieve(OCR_URL, archive)
    with gzip.open(archive, "rb") as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return path


def _read_words(path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    words: list[tuple[int, list[np.ndarray], list[int]]] = []
    current_pixels: list[np.ndarray] = []
    current_labels: list[int] = []
    current_fold = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 134:
                continue
            if not current_pixels:
                current_fold = int(parts[5])
            label = ord(parts[1]) - ord("a")
            pixels = np.asarray([float(value) for value in parts[6:134]], dtype=float)
            pixels = np.concatenate([pixels, np.ones(1, dtype=float)])
            current_pixels.append(pixels)
            current_labels.append(label)
            if int(parts[2]) == -1:
                words.append((
                    current_fold,
                    np.vstack(current_pixels),
                    np.asarray(current_labels, dtype=int),
                ))
                current_pixels = []
                current_labels = []
    return words
