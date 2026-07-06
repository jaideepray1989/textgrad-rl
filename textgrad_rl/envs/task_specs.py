"""Task specifications for generated ML-engineering repair repos."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    seed: int
    description: str
    files: dict[str, str]
    hidden_files: dict[str, str]
    visible_test_command: str
    hidden_validation_command: str
    train_command: str
    eval_command: str
    metric_name: str
    metric_threshold: float
    forbidden_paths: list[str] = field(default_factory=list)
    split: str = "train"

