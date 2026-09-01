from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

from .artifacts import artifact_root
from .config import load_config


def probe() -> dict[str, object]:
    import torch

    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "memory_gib": round(properties.total_memory / 1024**3, 2),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "artifact_root": str(root),
        "artifact_free_gib": round(shutil.disk_usage(root).free / 1024**3, 2),
        "hf_home": os.environ.get("HF_HOME"),
    }
    (root / "resource_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    for command in ("sft-fw", "rl-kl-fw"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    rlvr = subparsers.add_parser("rlvr")
    rlvr.add_argument("--config", required=True)
    rlvr.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.command == "probe":
        print(json.dumps(probe(), indent=2, sort_keys=True))
        return
    config = load_config(Path(args.config))
    if args.command == "sft-fw":
        from .training import train_sft_fw

        result = train_sft_fw(config)
    elif args.command == "rl-kl-fw":
        from .training import train_rl_kl_fw

        result = train_rl_kl_fw(config)
    else:
        from .rlvr import run_rlvr

        result = run_rlvr(config, args.variant)
    print(result)


if __name__ == "__main__":
    main()
