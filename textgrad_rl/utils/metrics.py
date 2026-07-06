"""Metric extraction and aggregation helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from textgrad_rl.types import ExperimentMetrics, Trajectory


def extract_json_metrics(text: str) -> dict[str, float]:
    """Extract the last JSON object containing numeric values from command output."""

    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in parsed.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    if metrics:
        return metrics
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([0-9.]+)", text):
        metrics[key] = float(value)
    return metrics


def aggregate_trajectories(
    trajectories: list[Trajectory],
    iteration: int,
    split: str,
    accepted_updates: int = 0,
    rejected_updates: int = 0,
    prompt_length_total: int = 0,
) -> ExperimentMetrics:
    if not trajectories:
        return ExperimentMetrics(
            iteration=iteration,
            split=split,
            success_rate=0.0,
            average_reward=0.0,
            test_pass_rate=0.0,
            invalid_action_rate=0.0,
            average_steps=0.0,
            command_count=0.0,
            accepted_updates=accepted_updates,
            rejected_updates=rejected_updates,
            runtime_seconds=0.0,
            prompt_length_total=prompt_length_total,
        )
    total_steps = sum(len(t.steps) for t in trajectories)
    invalid = sum(t.invalid_action_count for t in trajectories)
    tests_passed = sum(1 for t in trajectories if t.test_results.get("passed"))
    return ExperimentMetrics(
        iteration=iteration,
        split=split,
        success_rate=sum(1 for t in trajectories if t.success) / len(trajectories),
        average_reward=sum(t.total_reward for t in trajectories) / len(trajectories),
        test_pass_rate=tests_passed / len(trajectories),
        invalid_action_rate=invalid / max(total_steps, 1),
        average_steps=total_steps / len(trajectories),
        command_count=sum(t.command_count for t in trajectories) / len(trajectories),
        accepted_updates=accepted_updates,
        rejected_updates=rejected_updates,
        runtime_seconds=sum(t.runtime_seconds for t in trajectories),
        prompt_length_total=prompt_length_total,
    )


def metrics_to_row(metrics: ExperimentMetrics, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = metrics.__dict__.copy()
    if extra:
        row.update(extra)
    return row

