"""Evaluate initial heuristic prompts on a generated task split."""

from __future__ import annotations

import argparse
from pathlib import Path

from textgrad_rl.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed-prompt evaluation.")
    parser.add_argument("--output-dir", default="runs/evaluate_fixed_prompt")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_experiment(
        {
            "agent": "heuristic",
            "critic": "heuristic",
            "method": "fixed_prompt",
            "iterations": 0,
            "train_tasks": 0,
            "val_tasks": args.tasks,
            "test_tasks": args.tasks,
            "max_steps": 15,
            "seed": args.seed,
            "output_dir": str(Path(args.output_dir)),
            "command_timeout_sec": 10,
            "local_llm_base_url": None,
            "local_llm_model": None,
        }
    )


if __name__ == "__main__":
    main()

