#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${DBCFW_ARTIFACT_ROOT:-$HOME/.local/share/dbcfw/artifacts}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$artifact_root" "$hf_home"
cd "$package_root"

uv python install 3.11
uv venv --python 3.11 .venv
# The server driver exposes CUDA 12.9; pin the official cu128 build instead of
# letting PyPI select a CUDA 13 wheel that the installed driver cannot load.
uv pip install --python .venv/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[test]'

{
  printf 'export DBCFW_ARTIFACT_ROOT=%q\n' "$artifact_root"
  printf 'export HF_HOME=%q\n' "$hf_home"
  printf 'export TOKENIZERS_PARALLELISM=false\n'
  printf 'export PYTHONUNBUFFERED=1\n'
} > .env.server

source .env.server
.venv/bin/python -m pytest -q
.venv/bin/python -m dbcfw_llm.cli probe

printf 'Bootstrap complete. Artifacts: %s\n' "$DBCFW_ARTIFACT_ROOT"
