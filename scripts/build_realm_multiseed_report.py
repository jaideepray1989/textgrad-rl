"""Build clustered paired-bootstrap evidence for the REALM multi-seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


BOOTSTRAPS = 10_000
RNG_SEED = 20260709


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(fraction * len(values))))]


def paired_records(
    rows: list[dict[str, Any]],
    *,
    task_field: str,
    baseline_name: str,
    textgrad_name: str,
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row[task_field]), int(row["seed"]))][row["variant"]] = row
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (task, _), variants in sorted(by_key.items()):
        if baseline_name not in variants or textgrad_name not in variants:
            continue
        grouped[task].append((variants[baseline_name], variants[textgrad_name]))
    return grouped


def cluster_bootstrap(
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    statistic: Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float | None],
) -> tuple[float, float, float, int]:
    tasks = sorted(grouped)
    observed_pairs = [pair for task in tasks for pair in grouped[task]]
    observed = statistic(observed_pairs)
    if observed is None:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(RNG_SEED + len(observed_pairs) + len(tasks))
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for _ in tasks:
            task = tasks[rng.randrange(len(tasks))]
            task_pairs = grouped[task]
            sample.extend(task_pairs[rng.randrange(len(task_pairs))] for _ in task_pairs)
        value = statistic(sample)
        if value is not None:
            estimates.append(value)
    return observed, percentile(estimates, 0.025), percentile(estimates, 0.975), len(estimates)


def success_rate(which: int) -> Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float]:
    return lambda pairs: statistics.mean(float(bool(pair[which]["success"])) for pair in pairs)


def success_delta(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    return statistics.mean(float(bool(new["success"])) - float(bool(old["success"])) for old, new in pairs)


def common_success_step_decrease(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    median: bool,
) -> float | None:
    values = [
        float(old["turns"]) - float(new["turns"])
        for old, new in pairs
        if old["success"] and new["success"]
    ]
    if not values:
        return None
    return statistics.median(values) if median else statistics.mean(values)


def summarize(
    benchmark: str,
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
) -> dict[str, Any]:
    pairs = [pair for task in grouped for pair in grouped[task]]
    base = cluster_bootstrap(grouped, success_rate(0))
    textgrad = cluster_bootstrap(grouped, success_rate(1))
    delta = cluster_bootstrap(grouped, success_delta)
    mean_steps = cluster_bootstrap(
        grouped,
        lambda sample: common_success_step_decrease(sample, median=False),
    )
    median_steps = cluster_bootstrap(
        grouped,
        lambda sample: common_success_step_decrease(sample, median=True),
    )
    return {
        "benchmark": benchmark,
        "tasks": len(grouped),
        "paired_episodes": len(pairs),
        "seeds_per_task_min": min(len(grouped[task]) for task in grouped),
        "baseline_success": base[0],
        "baseline_success_ci_low": base[1],
        "baseline_success_ci_high": base[2],
        "textgrad_success": textgrad[0],
        "textgrad_success_ci_low": textgrad[1],
        "textgrad_success_ci_high": textgrad[2],
        "success_delta": delta[0],
        "success_delta_ci_low": delta[1],
        "success_delta_ci_high": delta[2],
        "common_success_pairs": sum(old["success"] and new["success"] for old, new in pairs),
        "mean_step_decrease": mean_steps[0],
        "mean_step_decrease_ci_low": mean_steps[1],
        "mean_step_decrease_ci_high": mean_steps[2],
        "median_step_decrease": median_steps[0],
        "median_step_decrease_ci_low": median_steps[1],
        "median_step_decrease_ci_high": median_steps[2],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_ci(row: dict[str, Any], prefix: str, signed: bool = False) -> str:
    value = 100 * row[prefix]
    low = 100 * row[f"{prefix}_ci_low"]
    high = 100 * row[f"{prefix}_ci_high"]
    fmt = "+.1f" if signed else ".1f"
    return f"{value:{fmt}}% [{low:{fmt}}, {high:{fmt}}]"


def delta_ci(row: dict[str, Any]) -> str:
    return (
        f"{100 * row['success_delta']:+.1f} pp "
        f"[{100 * row['success_delta_ci_low']:+.1f}, "
        f"{100 * row['success_delta_ci_high']:+.1f}]"
    )


def number_ci(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{row[prefix]:+.2f} "
        f"[{row[f'{prefix}_ci_low']:+.2f}, {row[f'{prefix}_ci_high']:+.2f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/realm_multiseed_qwen7b_t07_10seed")
    parser.add_argument("--report", default="REALM_MULTISEED_CONFIDENCE_INTERVALS.md")
    parser.add_argument("--textworld-label", default="TextWorld")
    parser.add_argument("--textworld-max-steps", type=int, default=80)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    textarena = paired_records(
        read_jsonl(run_dir / "textarena.jsonl"),
        task_field="env_id",
        baseline_name="no_textgrad",
        textgrad_name="textgrad_rl",
    )
    textworld = paired_records(
        read_jsonl(run_dir / "textworld.jsonl"),
        task_field="spec_id",
        baseline_name="qwen_ranker_base",
        textgrad_name="qwen_ranker_textgrad_rule",
    )
    textarena_row = summarize("TextArena", textarena)
    textworld_row = summarize(args.textworld_label, textworld)
    pooled_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, pairs in textarena.items():
        pooled_groups[f"textarena/{task}"].extend(pairs)
    for task, pairs in textworld.items():
        pooled_groups[f"textworld/{task}"].extend(pairs)
    rows = [textarena_row, textworld_row, summarize("Task-weighted aggregate", pooled_groups)]
    write_csv(run_dir / "paired_bootstrap_summary.csv", rows)

    lines = [
        "# REALM Multi-Seed Confidence Intervals",
        "",
        "Model: `qwen2.5:7b`; temperature: `0.7`. Intervals are 95% paired hierarchical bootstrap intervals (10,000 resamples), clustering by task and resampling seeds within task. TextWorldExpress is excluded.",
        "",
        "| Benchmark | Tasks x seeds | No TextGrad success | TextGrad-RL success | Paired improvement (pp) | Common successes | Mean step decrease | Median step decrease |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['tasks']} x {row['seeds_per_task_min']} | "
            f"{percent_ci(row, 'baseline_success')} | {percent_ci(row, 'textgrad_success')} | "
            f"{delta_ci(row)} | {row['common_success_pairs']} | "
            f"{number_ci(row, 'mean_step_decrease')} | {number_ci(row, 'median_step_decrease')} |"
        )
    lines.extend(
        [
            "",
            "Positive step decrease means TextGrad-RL used fewer actions. Step statistics include only paired episodes solved by both methods.",
            f"Macro-average success-rate improvement across the two benchmark rows: {100 * (textarena_row['success_delta'] + textworld_row['success_delta']) / 2:+.1f} percentage points.",
            "",
            "## Protocol",
            "",
            "- TextArena evaluates the previously train/validation-selected prompt policies on new held-out environment seeds.",
            f"- {args.textworld_label} evaluates the base and structured TextGrad controller rules with Qwen ranking controller-proposed actions and a {args.textworld_max_steps}-action cap.",
            "- Pairing uses identical task and environment seed across methods. Ollama does not expose a generation RNG seed through this runner, so paired rollouts do not use common model randomness.",
            "",
        ]
    )
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {run_dir / 'paired_bootstrap_summary.csv'}")


if __name__ == "__main__":
    main()
