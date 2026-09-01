from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    resolved = deepcopy(config)
    data = resolved.get("data")
    if isinstance(data, dict) and "path" in data:
        candidate = Path(data["path"])
        if not candidate.is_absolute():
            data["path"] = str((PACKAGE_ROOT / candidate).resolve())
    resolved["_source"] = str(config_path)
    return resolved
