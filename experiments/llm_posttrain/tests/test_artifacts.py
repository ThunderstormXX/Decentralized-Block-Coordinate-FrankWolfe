from __future__ import annotations

from pathlib import Path

from dbcfw_llm.artifacts import RunArtifacts


def test_artifacts_follow_external_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DBCFW_ARTIFACT_ROOT", str(tmp_path))
    run = RunArtifacts.create("unit", {"run_name": "smoke"})
    run.log({"step": 0})
    assert run.path.is_relative_to(tmp_path)
    assert run.metrics_path.read_text(encoding="utf-8").strip()
