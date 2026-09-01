#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"
source .env.server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

.venv/bin/python -m pytest -q
.venv/bin/python -m dbcfw_llm.cli probe

matrix_root="$DBCFW_ARTIFACT_ROOT/matrices/$(date -u +%Y%m%dT%H%M%SZ)_rlvr_pilot"
mkdir -p "$matrix_root"
: > "$matrix_root/run_paths.txt"

for variant in R0 R1 R2 R3 F1 F2 F3 F4; do
  .venv/bin/python -m dbcfw_llm.cli rlvr \
    --config configs/rlvr_pilot.yaml \
    --variant "$variant" | tee -a "$matrix_root/run_paths.txt"
  bash server/verify_clean_tree.sh
done

.venv/bin/python - "$matrix_root" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for raw in (root / "run_paths.txt").read_text().splitlines():
    run = Path(raw.strip())
    summary = json.loads((run / "final_summary.json").read_text())
    metrics_path = run / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    last = metrics[-1] if metrics else {}
    rows.append({
        "variant": summary.get("variant", "R0"),
        "pass_at_1_before": summary["eval_before"]["pass_at_1"],
        "pass_at_1_after": summary["eval_after"]["pass_at_1"],
        "generated_tokens_train": summary.get("generated_tokens_train", 0),
        "last_reward_mean": last.get("reward_mean"),
        "last_reverse_kl_k3": last.get("sampled_reverse_kl_k3_accepted"),
        "last_forward_kl_mean": last.get("exact_forward_token_kl_audit", {}).get("mean"),
        "last_dual_lambda": last.get("dual_lambda"),
        "last_peak_memory_gib": last.get("peak_memory_gib"),
        "run_path": str(run),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(root / "summary.json")
PY

bash server/verify_clean_tree.sh
