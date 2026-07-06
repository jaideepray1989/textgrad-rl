"""CLI entry point for TextGrad-RL experiments."""

from __future__ import annotations

from textgrad_rl.experiments.configs import build_parser, load_config_from_args
from textgrad_rl.experiments.runner import run_experiment


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config_from_args(args)
    output_dir = run_experiment(config)
    print(f"Run artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

