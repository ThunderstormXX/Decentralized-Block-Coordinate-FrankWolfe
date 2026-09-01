from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import PACKAGE_ROOT


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PACKAGE_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def artifact_root() -> Path:
    configured = os.environ.get("DBCFW_ARTIFACT_ROOT")
    root = Path(configured).expanduser() if configured else PACKAGE_ROOT / "artifacts"
    return root.resolve()


@dataclass
class RunArtifacts:
    path: Path
    metrics_path: Path

    @classmethod
    def create(cls, mode: str, config: dict[str, Any]) -> "RunArtifacts":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        name = str(config.get("run_name", mode)).replace("/", "-")
        run_path = artifact_root() / mode / f"{stamp}_{name}"
        run_path.mkdir(parents=True, exist_ok=False)
        with (run_path / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        environment = {
            "created_utc": stamp,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        (run_path / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        return cls(path=run_path, metrics_path=run_path / "metrics.jsonl")

    def log(self, payload: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.path / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
