from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .artifacts import RunArtifacts
from .data import arithmetic_batch, arithmetic_reward, read_sft_examples
from .kl import exact_forward_token_kl, kl_quantiles, sampled_reverse_kl_per_token
from .nuclear_adapter import NuclearFWLinear, adapters_disabled, inject_nuclear_adapters


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _load(config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    requested_device = model_cfg.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is false")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=_dtype(model_cfg.get("dtype", "bfloat16"))
    ).to(requested_device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter_cfg = config["adapter"]
    adapters = inject_nuclear_adapters(
        model,
        adapter_cfg["target_suffixes"],
        max_modules=adapter_cfg.get("max_modules"),
    )
    return model, tokenizer, adapters, torch.device(requested_device)


def _gamma(config_value: Any, step: int) -> float:
    if config_value == "line_search_schedule":
        return 2.0 / float(step + 2)
    return float(config_value)


def _adapter_summary(adapters: dict[str, NuclearFWLinear]) -> dict[str, Any]:
    return {
        name: {
            "rank": adapter.numerical_rank(),
            "nuclear_norm": adapter.nuclear_norm(),
            "atoms": len(adapter.atoms),
        }
        for name, adapter in adapters.items()
    }


@torch.no_grad()
def _calibration(model, tokenizer, adapters, prefixes: list[str], device: torch.device) -> dict[str, float]:
    encoded = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True).to(device)
    policy_logits = model(**encoded).logits
    with adapters_disabled(adapters):
        reference_logits = model(**encoded).logits
    mask = encoded["attention_mask"].bool()
    values = exact_forward_token_kl(reference_logits, policy_logits)[mask]
    return kl_quantiles(values)


def _save_adapter_checkpoint(adapters: dict[str, NuclearFWLinear], path: Path) -> None:
    payload = {f"{name}.delta": adapter.delta.detach().cpu() for name, adapter in adapters.items()}
    torch.save(payload, path)


def train_sft_fw(config: dict[str, Any]) -> Path:
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    artifacts = RunArtifacts.create("sft-fw", config)
    model, tokenizer, adapters, device = _load(config)
    examples = read_sft_examples(config["data"]["path"])
    training_cfg = config["training"]
    radius = float(config["adapter"]["radius"])
    lmo_iters = int(config["adapter"].get("lmo_iters", 8))
    model.train()
    for step in range(int(training_cfg["steps"])):
        started = time.perf_counter()
        batch = examples[step % len(examples)]
        prompt_ids = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        full = tokenizer(
            batch["prompt"] + batch["response"] + tokenizer.eos_token,
            return_tensors="pt",
            truncation=True,
            max_length=int(training_cfg["max_length"]),
        ).to(device)
        labels = full["input_ids"].clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        output = model(**full, labels=labels)
        output.loss.backward()
        gamma = _gamma(training_cfg["gamma"], step)
        lmo = {
            name: adapter.fw_update(radius=radius, gamma=gamma, lmo_iters=lmo_iters)
            for name, adapter in adapters.items()
        }
        calibration = _calibration(
            model, tokenizer, adapters, list(config["calibration"]["prefixes"]), device
        )
        artifacts.log(
            {
                "step": step,
                "train_loss": float(output.loss.detach().item()),
                "exact_forward_token_kl": calibration,
                "adapters": _adapter_summary(adapters),
                "lmo": lmo,
                "wall_time_sec": time.perf_counter() - started,
            }
        )
    _save_adapter_checkpoint(adapters, artifacts.path / "adapter.pt")
    artifacts.write_json("final_summary.json", {"adapters": _adapter_summary(adapters)})
    return artifacts.path


def _sequence_logps(model, input_ids, prompt_length: int) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits[:, :-1]
    targets = input_ids[:, 1:]
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_length - 1)
    return token_logps[:, start:]


@torch.no_grad()
def _sampled_kl_on_sequences(model, samples: list[dict[str, Any]]) -> float:
    values = []
    for sample in samples:
        policy_logps = _sequence_logps(model, sample["input_ids"], sample["prompt_length"])
        values.append(
            sampled_reverse_kl_per_token(policy_logps, sample["reference_logps"]).mean()
        )
    return float(torch.stack(values).mean().item())


def train_rl_kl_fw(config: dict[str, Any]) -> Path:
    torch.manual_seed(int(config["seed"]))
    rng = random.Random(int(config["seed"]))
    artifacts = RunArtifacts.create("rl-kl-fw", config)
    model, tokenizer, adapters, device = _load(config)
    train_cfg = config["training"]
    constraint = config["constraint"]
    radius = float(config["adapter"]["radius"])
    lmo_iters = int(config["adapter"].get("lmo_iters", 8))
    dual_lambda = float(constraint["initial_lambda"])
    target = float(constraint["sampled_reverse_kl_target_per_token"])
    model.train()
    for step in range(int(train_cfg["steps"])):
        started = time.perf_counter()
        rows = arithmetic_batch(rng, int(train_cfg["batch_size"]))
        losses = []
        sampled_kls = []
        rewards = []
        generations = []
        sampled_sequences = []
        for row in rows:
            prompt = str(row["prompt"])
            prompt_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(train_cfg["max_prompt_length"]),
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **prompt_tokens,
                    do_sample=True,
                    temperature=float(train_cfg["temperature"]),
                    max_new_tokens=int(train_cfg["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = int(prompt_tokens["input_ids"].shape[1])
            response_ids = generated[:, prompt_length:]
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            reward = arithmetic_reward(response, int(row["answer"]))
            policy_logps = _sequence_logps(model, generated, prompt_length)
            with torch.no_grad(), adapters_disabled(adapters):
                reference_logps = _sequence_logps(model, generated, prompt_length)
            reverse_kl = sampled_reverse_kl_per_token(policy_logps, reference_logps)
            advantage = torch.tensor(reward - 0.10, device=device)
            losses.append(-(advantage * policy_logps.mean()) + dual_lambda * reverse_kl.mean())
            sampled_kls.append(reverse_kl.detach().mean())
            sampled_sequences.append(
                {
                    "input_ids": generated,
                    "prompt_length": prompt_length,
                    "reference_logps": reference_logps,
                }
            )
            rewards.append(reward)
            generations.append({"prompt": prompt, "response": response, "reward": reward})
        loss = torch.stack(losses).mean()
        loss.backward()
        base_states = {name: adapter.delta.detach().clone() for name, adapter in adapters.items()}
        proposals = {
            name: adapter.propose_atom(lmo_iters=lmo_iters) for name, adapter in adapters.items()
        }
        gamma = float(train_cfg["gamma"])
        backtracks = 0
        model.eval()
        while True:
            for name, adapter in adapters.items():
                with torch.no_grad():
                    adapter.delta.copy_(base_states[name])
                left, right, _ = proposals[name]
                adapter.apply_atom(left, right, radius, gamma, record=False)
            candidate_kl = _sampled_kl_on_sequences(model, sampled_sequences)
            if candidate_kl <= target * (1.0 + float(constraint["tolerance"])):
                break
            backtracks += 1
            if backtracks > int(constraint["max_backtracks"]):
                for name, adapter in adapters.items():
                    with torch.no_grad():
                        adapter.delta.copy_(base_states[name])
                gamma = 0.0
                candidate_kl = _sampled_kl_on_sequences(model, sampled_sequences)
                break
            gamma *= 0.5
        if gamma > 0.0:
            for name, adapter in adapters.items():
                with torch.no_grad():
                    adapter.delta.copy_(base_states[name])
                left, right, _ = proposals[name]
                adapter.apply_atom(left, right, radius, gamma, record=True)
        for adapter in adapters.values():
            adapter.delta.grad = None
        model.train()
        pre_step_kl = float(torch.stack(sampled_kls).mean().item())
        mean_kl = candidate_kl
        dual_lambda = max(0.0, dual_lambda + float(constraint["dual_lr"]) * (mean_kl - target))
        calibration = _calibration(
            model, tokenizer, adapters, list(config["calibration"]["prefixes"]), device
        )
        artifacts.log(
            {
                "step": step,
                "policy_loss": float(loss.detach().item()),
                "mean_reward": sum(rewards) / len(rewards),
                "sampled_reverse_kl_per_token": mean_kl,
                "sampled_reverse_kl_pre_step": pre_step_kl,
                "kl_target": target,
                "dual_lambda": dual_lambda,
                "gamma_accepted": gamma,
                "backtracks": backtracks,
                "exact_forward_token_kl_audit": calibration,
                "adapters": _adapter_summary(adapters),
                "generations": generations,
                "wall_time_sec": time.perf_counter() - started,
            }
        )
    _save_adapter_checkpoint(adapters, artifacts.path / "adapter.pt")
    artifacts.write_json(
        "final_summary.json", {"dual_lambda": dual_lambda, "adapters": _adapter_summary(adapters)}
    )
    return artifacts.path
