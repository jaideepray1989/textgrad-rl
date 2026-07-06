"""Configuration loading for experiment CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from textgrad_rl.utils.json_utils import read_json


DEFAULT_CONFIG: dict[str, Any] = {
    "agent": "heuristic",
    "critic": "heuristic",
    "method": "modular_textgrad",
    "iterations": 2,
    "train_tasks": 8,
    "val_tasks": 4,
    "test_tasks": 4,
    "max_steps": 15,
    "seed": 7,
    "output_dir": "runs/mac_demo",
    "command_timeout_sec": 10,
    "local_llm_base_url": None,
    "local_llm_model": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextGrad-RL experiments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--agent", choices=["heuristic", "local_llm"], default=None)
    parser.add_argument("--critic", choices=["heuristic", "local_llm"], default=None)
    parser.add_argument(
        "--method",
        choices=[
            "fixed_prompt",
            "scalar_prompt_search",
            "modular_textgrad",
            "monolithic_textgrad",
            "no_acceptance_gate",
        ],
        default=None,
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--train-tasks", type=int, default=None)
    parser.add_argument("--val-tasks", type=int, default=None)
    parser.add_argument("--test-tasks", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--local-llm-base-url", type=str, default=None)
    parser.add_argument("--local-llm-model", type=str, default=None)
    return parser


def load_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(read_json(Path(args.config)))
    cli_to_key = {
        "agent": "agent",
        "critic": "critic",
        "method": "method",
        "iterations": "iterations",
        "train_tasks": "train_tasks",
        "val_tasks": "val_tasks",
        "test_tasks": "test_tasks",
        "max_steps": "max_steps",
        "seed": "seed",
        "output_dir": "output_dir",
        "local_llm_base_url": "local_llm_base_url",
        "local_llm_model": "local_llm_model",
    }
    for attr, key in cli_to_key.items():
        value = getattr(args, attr, None)
        if value is not None:
            config[key] = value
    return config

