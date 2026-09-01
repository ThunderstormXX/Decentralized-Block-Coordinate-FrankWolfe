#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
status="$(git status --porcelain)"
if [[ -n "$status" ]]; then
  printf '%s\n' "$status"
  printf 'Git tree is not clean after the run.\n' >&2
  exit 1
fi
printf 'Git tree is clean; run artifacts are outside the checkout or ignored.\n'
