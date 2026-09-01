#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"
source .env.server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

.venv/bin/python -m pytest -q
.venv/bin/python -m dbcfw_llm.cli probe
bash server/run_sft_fw_smoke.sh
bash server/verify_clean_tree.sh
bash server/run_rl_kl_fw_smoke.sh
bash server/verify_clean_tree.sh
