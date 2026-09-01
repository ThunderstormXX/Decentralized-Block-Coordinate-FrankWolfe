# LLM post-training smoke harness

This package starts the empirical LLM stage without mixing it with the existing
convex benchmark. It contains two deliberately small end-to-end paths:

- `sft-fw`: frozen Hugging Face causal LM plus direct nuclear-norm adapters,
  updated with rank-one Frank-Wolfe LMOs;
- `rl-kl-fw`: the same adapter geometry with a primal-dual sampled reverse-KL
  constraint and exact forward-token-KL audit.
- `rlvr`: a common Dr. GRPO-style GSM8K protocol for the gated R0–F4 baseline
  matrix (frozen, AdamW/LoRA, nuclear FW, penalty, dual and backtracking).

These are non-convex/stochastic experiments. They do not inherit the convex
DBCFW convergence guarantee.

## Artifact policy

Every run goes to `DBCFW_ARTIFACT_ROOT`, normally an absolute directory outside
the checkout. If the variable is absent, the fallback is the ignored local
`artifacts/` directory. A run writes its resolved config, environment snapshot,
JSONL metrics and adapter checkpoint into a timestamped directory.

On `opt_2`, bootstrap creates:

```text
~/DBCFW                              # clean Git checkout
~/.local/share/dbcfw/artifacts       # metrics, checkpoints, generated text
~/DBCFW/experiments/llm_posttrain/.env.server  # ignored local settings
```

## Server bootstrap

```bash
cd ~/DBCFW/experiments/llm_posttrain
bash server/bootstrap_opt2.sh
bash server/run_end_to_end.sh
bash server/run_rlvr_pilot_matrix.sh
```

The server scripts currently default to CUDA ordinal 6, the available RTX on
the shared `opt_2` host; override `CUDA_VISIBLE_DEVICES` when scheduling changes.
Smoke configs use `float16`, which runs on both the RTX fallback and A100s.
Bootstrap pins the official PyTorch 2.8 CUDA 12.8 wheel because the server's
CUDA 12.9-capable driver cannot load CUDA 13 wheels.

`run_end_to_end.sh` is gated with `set -e`: tests and the resource probe must
pass before SFT starts, SFT and the clean-tree check must pass before the
KL-constrained RL smoke starts, and a final clean-tree check closes the run.

## Direct commands

```bash
source .env.server
.venv/bin/python -m dbcfw_llm.cli probe
.venv/bin/python -m dbcfw_llm.cli sft-fw --config configs/sft_fw_smoke.yaml
.venv/bin/python -m dbcfw_llm.cli rl-kl-fw --config configs/rl_kl_fw_smoke.yaml
```

The smoke fixtures are project-authored arithmetic/instruction examples. The
default checkpoint is `Qwen/Qwen2.5-0.5B`, whose model card reports an Apache-2.0
license. Model downloads remain in the external Hugging Face cache.

The 0.5B matrix is an implementation pilot, not a reasoning result. The
pre-registered research configuration is `configs/rlvr_research_15b.yaml`; it
must only be interpreted after a longer token-matched run with multiple seeds.
