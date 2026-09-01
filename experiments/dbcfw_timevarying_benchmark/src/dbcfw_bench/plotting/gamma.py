from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def gamma_plot(frame: pd.DataFrame, path: Path) -> Path:
    if "gamma" in frame.columns:
        data = frame[["iteration", "gamma"]].dropna().drop_duplicates()
    else:
        data = frame[["iteration"]].drop_duplicates()
        data = data[data["iteration"] > 0].copy()
        data["gamma"] = 2.0 / (data["iteration"] + 1.0)
    data = data.sort_values("iteration")
    plt.figure(figsize=(7, 4))
    plt.plot(data["iteration"], data["gamma"], marker="o", markersize=2)
    plt.xlabel("iteration")
    plt.ylabel("gamma_t")
    plt.title("Frank-Wolfe step size: gamma_t = 2/(t+2), zero-based t")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
