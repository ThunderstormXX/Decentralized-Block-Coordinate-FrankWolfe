from __future__ import annotations

import json
import random
import re
from pathlib import Path


def read_sft_examples(path: str | Path) -> list[dict[str, str]]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                examples.append({"prompt": str(row["prompt"]), "response": str(row["response"])})
    if not examples:
        raise ValueError("SFT fixture contains no examples")
    return examples


def arithmetic_batch(rng: random.Random, size: int) -> list[dict[str, object]]:
    rows = []
    for _ in range(size):
        left = rng.randint(1, 20)
        right = rng.randint(1, 20)
        operation = rng.choice(["+", "-"])
        answer = left + right if operation == "+" else left - right
        rows.append(
            {
                "prompt": f"Question: What is {left} {operation} {right}? Answer with only the integer:\n",
                "answer": answer,
            }
        )
    return rows


def arithmetic_reward(text: str, answer: int) -> float:
    matches = re.findall(r"[-+]?\d+", text)
    if not matches:
        return 0.0
    prediction = int(matches[0])
    if prediction == answer:
        return 1.0
    return 0.25 / (1.0 + abs(prediction - answer))
