"""Task factory and deterministic train/val/test split creation."""

from __future__ import annotations

import itertools
import random

from textgrad_rl.envs.task_specs import TaskSpec
from textgrad_rl.envs.task_templates import TEMPLATE_BUILDERS


FAMILIES = list(TEMPLATE_BUILDERS)


DESCRIPTIONS = {
    "shape_mismatch_training_crash": (
        "Training crashes because preprocessing/model code assumes a wrong feature dimension. "
        "Fix the source code so smoke tests, training, and eval complete."
    ),
    "missing_column_preprocessing": (
        "Preprocessing expects a stale column name that is absent from the current CSV schema. "
        "Make schema handling robust without editing tests or data."
    ),
    "reproducibility_failure": (
        "The training/evaluation loop is nondeterministic. Set deterministic random state usage "
        "so repeated metrics are reproducible."
    ),
    "metric_regression": (
        "The pipeline runs but evaluation accuracy is below the required threshold. Diagnose the "
        "metric regression and fix the underlying source-code bug."
    ),
    "inference_latency_regression": (
        "Batch inference is correct but too slow because unnecessary work happens per prediction. "
        "Keep predictions correct while reducing latency."
    ),
}


def create_task_spec(family: str, seed: int, split: str = "train", index: int = 0) -> TaskSpec:
    if family not in TEMPLATE_BUILDERS:
        raise ValueError(f"Unknown task family: {family}")
    files, hidden_files, metric_name, threshold = TEMPLATE_BUILDERS[family](seed)
    forbidden_paths = sorted(
        set(hidden_files)
        | {
            "task_metadata.json",
            "reward.py",
            "tests/test_training.py",
            "tests/test_preprocess.py",
            "tests/test_reproducibility.py",
            "tests/test_metric.py",
            "tests/test_latency.py",
        }
    )
    return TaskSpec(
        task_id=f"{split}_{index:03d}_{family}_seed_{seed}",
        family=family,
        seed=seed,
        split=split,
        description=DESCRIPTIONS[family],
        files=files,
        hidden_files=hidden_files,
        visible_test_command="pytest -q",
        hidden_validation_command="python hidden_validation.py",
        train_command="python train.py",
        eval_command="python eval.py",
        metric_name=metric_name,
        metric_threshold=threshold,
        forbidden_paths=forbidden_paths,
    )


def build_tasks(count: int, seed: int, split: str) -> list[TaskSpec]:
    rng = random.Random(seed)
    families = list(itertools.islice(itertools.cycle(FAMILIES), count))
    rng.shuffle(families)
    return [
        create_task_spec(family, seed + i * 997 + len(split), split=split, index=i)
        for i, family in enumerate(families)
    ]


def build_task_splits(
    train_tasks: int,
    val_tasks: int,
    test_tasks: int,
    seed: int,
) -> tuple[list[TaskSpec], list[TaskSpec], list[TaskSpec]]:
    return (
        build_tasks(train_tasks, seed + 11, "train"),
        build_tasks(val_tasks, seed + 101, "val"),
        build_tasks(test_tasks, seed + 1001, "test"),
    )

