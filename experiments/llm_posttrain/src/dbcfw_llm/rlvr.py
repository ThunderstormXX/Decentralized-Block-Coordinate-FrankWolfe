from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .artifacts import RunArtifacts
from .kl import exact_forward_token_kl, kl_quantiles, sampled_reverse_kl_k3
from .lora_adapter import LoRALinear, inject_lora_adapters
from .nuclear_adapter import NuclearFWLinear, adapters_disabled, inject_nuclear_adapters


def extract_final_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
    hashes = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    matches = hashes or boxed or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def exact_answer_reward(completion: str, answer: str) -> float:
    predicted = extract_final_number(completion)
    target = extract_final_number(answer)
    if predicted is None or target is None:
        return 0.0
    try:
        return float(Decimal(predicted) == Decimal(target))
    except InvalidOperation:
        return 0.0


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("Dr. GRPO requires at least two rewards in a group")
    return rewards - rewards.mean()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _variant_config(config: dict[str, Any], variant_name: str) -> dict[str, Any]:
    if variant_name not in config["variants"]:
        raise KeyError(f"Unknown variant {variant_name!r}")
    result = deepcopy(config)
    variant = result.pop("variants")[variant_name]
    result["variant"] = variant_name
    result["method"] = variant
    result["run_name"] = f"{result['run_name']}_{variant_name}"
    return result


def _load_model(config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    device = torch.device(model_cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=_dtype(model_cfg.get("dtype", "float16"))
    ).to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    method = config["method"]
    adapter_cfg = config["adapter"]
    if method["geometry"] == "lora":
        adapters = inject_lora_adapters(
            model,
            adapter_cfg["target_suffixes"],
            rank=int(adapter_cfg["lora_rank"]),
            alpha=float(adapter_cfg["lora_alpha"]),
            max_modules=adapter_cfg.get("max_modules"),
        )
    elif method["geometry"] == "nuclear_fw":
        adapters = inject_nuclear_adapters(
            model,
            adapter_cfg["target_suffixes"],
            max_modules=adapter_cfg.get("max_modules"),
        )
    elif method["geometry"] == "frozen":
        adapters = {}
    else:
        raise ValueError(f"Unsupported geometry: {method['geometry']}")
    return model, tokenizer, adapters, device


def _load_gsm8k(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main")
    seed = int(config["seed"])
    train = data["train"].shuffle(seed=seed)
    test = data["test"].shuffle(seed=seed + 1)
    train_rows = [dict(row) for row in train.select(range(int(config["data"]["train_samples"])))]
    eval_rows = [dict(row) for row in test.select(range(int(config["data"]["eval_samples"])))]
    return train_rows, eval_rows


def _prompt(question: str) -> str:
    return (
        "Solve the grade-school math problem. Show concise reasoning and put the final "
        f"numeric answer after ####.\nQuestion: {question}\nAnswer:"
    )


def _completion_mask(response_ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    if eos_token_id is None:
        return torch.ones_like(response_ids, dtype=torch.bool)
    eos = response_ids.eq(eos_token_id)
    return eos.cumsum(dim=1).le(1)


def _completion_logps(model, sequences: torch.Tensor, prompt_length: int) -> torch.Tensor:
    logits = model(input_ids=sequences).logits[:, :-1]
    targets = sequences[:, 1:]
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return token_logps[:, max(0, prompt_length - 1) :]


@torch.no_grad()
def _calibration(model, tokenizer, adapters, prompts: list[str], device) -> dict[str, float]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    policy_logits = model(**encoded).logits
    with adapters_disabled(adapters):
        reference_logits = model(**encoded).logits
    values = exact_forward_token_kl(reference_logits, policy_logits)[encoded["attention_mask"].bool()]
    return kl_quantiles(values)


@torch.no_grad()
def _evaluate(model, tokenizer, adapters, rows, device, max_new_tokens: int) -> dict[str, float]:
    del adapters
    correct = 0.0
    generated_tokens = 0
    model.eval()
    for row in rows:
        encoded = tokenizer(_prompt(row["question"]), return_tensors="pt").to(device)
        sequence = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        response_ids = sequence[:, encoded["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        correct += exact_answer_reward(response, row["answer"])
        generated_tokens += int(response_ids.numel())
    return {"pass_at_1": correct / len(rows), "generated_tokens": generated_tokens}


def _adapter_summary(adapters: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, adapter in adapters.items():
        result[name] = {
            "rank": adapter.numerical_rank(),
            "nuclear_norm": adapter.nuclear_norm(),
            "trainable_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        }
        if isinstance(adapter, NuclearFWLinear):
            result[name]["atoms"] = len(adapter.atoms)
    return result


def _save_checkpoint(adapters: dict[str, Any], path: Path) -> None:
    payload = {}
    for name, adapter in adapters.items():
        if isinstance(adapter, NuclearFWLinear):
            payload[f"{name}.delta"] = adapter.delta.detach().cpu()
        elif isinstance(adapter, LoRALinear):
            payload[f"{name}.lora_a"] = adapter.lora_a.detach().cpu()
            payload[f"{name}.lora_b"] = adapter.lora_b.detach().cpu()
    torch.save(payload, path)


@torch.no_grad()
def _candidate_kl(model, samples: list[dict[str, Any]]) -> float:
    values = []
    for sample in samples:
        policy = _completion_logps(model, sample["sequences"], sample["prompt_length"])
        kl = sampled_reverse_kl_k3(policy, sample["reference_logps"])
        values.append(kl[sample["mask"]].mean())
    return float(torch.stack(values).mean().item())


def run_rlvr(config: dict[str, Any], variant_name: str) -> Path:
    run = _variant_config(config, variant_name)
    seed = int(run["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    artifacts = RunArtifacts.create("rlvr", run)
    model, tokenizer, adapters, device = _load_model(run)
    train_rows, eval_rows = _load_gsm8k(run)
    method = run["method"]
    training = run["training"]
    constraint = run["constraint"]
    calibration_prompts = [_prompt(row["question"]) for row in eval_rows]
    before = _evaluate(
        model, tokenizer, adapters, eval_rows, device, int(training["eval_max_new_tokens"])
    )
    artifacts.write_json("eval_before.json", before)
    if method["optimizer"] == "none":
        artifacts.write_json("final_summary.json", {"eval_before": before, "eval_after": before})
        return artifacts.path

    optimizer = None
    if method["optimizer"] == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(training["learning_rate"]), weight_decay=0.0
        )
    dual_lambda = float(constraint["initial_lambda"])
    target = float(constraint["target_per_token"])
    total_generated_tokens = 0
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    model.train()
    for step in range(int(training["steps"])):
        started = time.perf_counter()
        row = train_rows[step % len(train_rows)]
        prompt = _prompt(row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            sequences = model.generate(
                **encoded,
                do_sample=True,
                temperature=float(training["temperature"]),
                top_p=float(training["top_p"]),
                num_return_sequences=int(training["group_size"]),
                max_new_tokens=int(training["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
            )
        response_ids = sequences[:, prompt_length:]
        mask = _completion_mask(response_ids, tokenizer.eos_token_id)
        completions = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        rewards = torch.tensor(
            [exact_answer_reward(text, row["answer"]) for text in completions],
            dtype=torch.float32,
            device=device,
        )
        advantages = group_advantages(rewards)
        policy_logps = _completion_logps(model, sequences, prompt_length)
        with torch.no_grad(), adapters_disabled(adapters):
            reference_logps = _completion_logps(model, sequences, prompt_length)
        sampled_kl = sampled_reverse_kl_k3(policy_logps, reference_logps)
        pg_loss = -(
            advantages[:, None] * policy_logps * mask
        ).sum() / (float(training["group_size"]) * float(training["max_new_tokens"]))
        kl_mean = sampled_kl[mask].mean()
        control = method["kl_control"]
        coefficient = 0.0
        if control == "penalty":
            coefficient = float(method["beta"])
        elif control == "dual":
            coefficient = dual_lambda
        loss = pg_loss + coefficient * kl_mean
        loss.backward()
        pre_step_kl = float(kl_mean.detach().item())
        accepted_gamma = None
        backtracks = 0
        accepted_kl = pre_step_kl
        lmo_stats = {}
        if method["optimizer"] == "adamw":
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                accepted_kl = _candidate_kl(
                    model,
                    [{"sequences": sequences, "prompt_length": prompt_length, "reference_logps": reference_logps, "mask": mask}],
                )
        else:
            radius = float(run["adapter"]["radius"])
            lmo_iters = int(run["adapter"]["lmo_iters"])
            gamma = float(training["fw_gamma"])
            if control == "backtracking":
                base = {name: adapter.delta.detach().clone() for name, adapter in adapters.items()}
                proposals = {
                    name: adapter.propose_atom(lmo_iters=lmo_iters)
                    for name, adapter in adapters.items()
                }
                model.eval()
                while True:
                    for name, adapter in adapters.items():
                        with torch.no_grad():
                            adapter.delta.copy_(base[name])
                        left, right, _ = proposals[name]
                        adapter.apply_atom(left, right, radius, gamma, record=False)
                    accepted_kl = _candidate_kl(
                        model,
                        [{"sequences": sequences, "prompt_length": prompt_length, "reference_logps": reference_logps, "mask": mask}],
                    )
                    if accepted_kl <= target * (1.0 + float(constraint["tolerance"])):
                        break
                    backtracks += 1
                    if backtracks > int(constraint["max_backtracks"]):
                        gamma = 0.0
                        for name, adapter in adapters.items():
                            with torch.no_grad():
                                adapter.delta.copy_(base[name])
                        accepted_kl = _candidate_kl(
                            model,
                            [{"sequences": sequences, "prompt_length": prompt_length, "reference_logps": reference_logps, "mask": mask}],
                        )
                        break
                    gamma *= 0.5
                if gamma > 0:
                    for name, adapter in adapters.items():
                        with torch.no_grad():
                            adapter.delta.copy_(base[name])
                        left, right, sigma = proposals[name]
                        adapter.apply_atom(left, right, radius, gamma, record=True)
                        lmo_stats[name] = {"sigma": sigma, "iterations": lmo_iters}
                accepted_gamma = gamma
                for adapter in adapters.values():
                    adapter.delta.grad = None
                model.train()
            else:
                accepted_gamma = gamma
                for name, adapter in adapters.items():
                    lmo_stats[name] = adapter.fw_update(radius, gamma, lmo_iters)
                accepted_kl = _candidate_kl(
                    model,
                    [{"sequences": sequences, "prompt_length": prompt_length, "reference_logps": reference_logps, "mask": mask}],
                )
        if control == "dual":
            dual_lambda = max(
                0.0,
                dual_lambda + float(constraint["dual_lr"]) * (accepted_kl - target),
            )
        generated = int(mask.sum().item())
        total_generated_tokens += generated
        calibration = _calibration(model, tokenizer, adapters, calibration_prompts, device)
        artifacts.log(
            {
                "step": step,
                "variant": variant_name,
                "reward_mean": float(rewards.mean().item()),
                "reward_std": float(rewards.std(unbiased=False).item()),
                "zero_reward_variance": bool(rewards.std(unbiased=False).item() == 0.0),
                "policy_loss": float(pg_loss.detach().item()),
                "sampled_reverse_kl_k3_pre_step": pre_step_kl,
                "sampled_reverse_kl_k3_accepted": accepted_kl,
                "kl_target": target,
                "kl_coefficient": coefficient,
                "dual_lambda": dual_lambda,
                "accepted_gamma": accepted_gamma,
                "backtracks": backtracks,
                "exact_forward_token_kl_audit": calibration,
                "generated_tokens_step": generated,
                "generated_tokens_total": total_generated_tokens,
                "adapters": _adapter_summary(adapters),
                "lmo": lmo_stats,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
                ),
                "wall_time_sec": time.perf_counter() - started,
            }
        )
    after = _evaluate(
        model, tokenizer, adapters, eval_rows, device, int(training["eval_max_new_tokens"])
    )
    _save_checkpoint(adapters, artifacts.path / "adapter.pt")
    artifacts.write_json(
        "final_summary.json",
        {
            "variant": variant_name,
            "eval_before": before,
            "eval_after": after,
            "generated_tokens_train": total_generated_tokens,
            "dual_lambda": dual_lambda,
            "adapters": _adapter_summary(adapters),
        },
    )
    return artifacts.path


def summarize_matrix(root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.glob("*/final_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": summary_path.parent.name,
                "variant": summary.get("variant", "R0"),
                "pass_at_1_before": summary["eval_before"]["pass_at_1"],
                "pass_at_1_after": summary["eval_after"]["pass_at_1"],
                "generated_tokens_train": summary.get("generated_tokens_train", 0),
            }
        )
    return rows
