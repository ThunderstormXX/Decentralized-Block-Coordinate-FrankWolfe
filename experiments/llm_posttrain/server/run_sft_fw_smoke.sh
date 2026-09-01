#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"
source .env.server
# CUDA ordinal 6 is currently the least-loaded device that accepts new contexts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
.venv/bin/python -m dbcfw_llm.cli sft-fw --config configs/sft_fw_smoke.yaml
