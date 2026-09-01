#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"
source .env.server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
.venv/bin/python -m dbcfw_llm.cli rl-kl-fw --config configs/rl_kl_fw_smoke.yaml
