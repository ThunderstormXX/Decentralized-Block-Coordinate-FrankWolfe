from __future__ import annotations

from pathlib import Path

SKIP_TOKENS = ("smoke", "probe", "muon")
ARTIFACT_RUNS = {"grid", "high_batch_sanity", "nblocks_showcase", "showcase", "single"}
LMO_PREFIXES = ("l1_block_", "l2_block_", "box_")
LMO_ORDER = {"box": 0, "l1_block": 1, "l2_block": 2}


def result_files(runs_dirs) -> list[Path]:
    files = []
    for root in _roots(runs_dirs):
        for path in sorted(root.glob("*/results.csv")):
            name = path.parent.name.lower()
            if name not in ARTIFACT_RUNS and not any(t in name for t in SKIP_TOKENS):
                files.append(path)
    return files


def display_name(name: str, lmo: str) -> str:
    if lmo == "box":
        return name
    for prefix in LMO_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def grouped_sort_key(name: str, lmo: str) -> tuple[str, int, str]:
    base = display_name(name, lmo)
    return base, LMO_ORDER.get(lmo, 99), name


def _roots(runs_dirs) -> list[Path]:
    if isinstance(runs_dirs, (str, Path)):
        return [Path(part) for part in str(runs_dirs).split(",") if part]
    return [Path(path) for path in runs_dirs]
