"""Optional matplotlib plots for metrics CSV files."""

from __future__ import annotations

from pathlib import Path


def plot_metrics_csv(metrics_csv: Path, output_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return False
    if not metrics_csv.exists():
        return False
    df = pd.read_csv(metrics_csv)
    if df.empty or "iteration" not in df or "success_rate" not in df:
        return False
    output_png.parent.mkdir(parents=True, exist_ok=True)
    for split, group in df.groupby("split"):
        plt.plot(group["iteration"], group["success_rate"], marker="o", label=split)
    plt.xlabel("Iteration")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    return True

