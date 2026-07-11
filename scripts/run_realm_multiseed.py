"""Run resumable multi-seed REALM evaluations for TextArena and TextWorld."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_textworld24_structured_probe import VARIANT_CONFIGS, suite_record
from scripts.run_textworld_simple1_structured_probe import run_structured_variant
from textgrad_rl.benchmarks.textarena_expanded_suites import (
    PUZZLE_SLM_ENVS,
    REAL_SLM_ENVS,
    SOCIAL_SLM_ENVS,
    OpenAICompatibleChatModel,
    initial_slm_variables,
    run_slm_episode,
)
from textgrad_rl.benchmarks.textworld_24_suite import default_specs, ensure_games
from textgrad_rl.types import TextVariable
from textgrad_rl.utils.json_utils import append_jsonl


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEXTARENA_RUN = ROOT / "runs" / "qwen25_7b_full_textgames" / "textarena"
TEXTARENA_SUITES = {
    "puzzle": (PUZZLE_SLM_ENVS, "puzzle_slm"),
    "social": (SOCIAL_SLM_ENVS, "social_slm"),
    "real_slm": (REAL_SLM_ENVS, "real_slm"),
}
TEXTWORLD_VARIANTS = ("qwen_ranker_base", "qwen_ranker_textgrad_rule")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_variables(path: Path) -> dict[str, TextVariable]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: TextVariable(**value) for name, value in raw.items()}


def run_textarena(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "textarena.jsonl"
    completed = {
        (row["suite"], row["env_id"], int(row["seed"]), row["variant"])
        for row in read_jsonl(path)
    }
    model = OpenAICompatibleChatModel(
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=64,
        timeout=args.request_timeout,
    )
    total_envs = sum(len(envs[: args.textarena_limit or None]) for envs, _ in TEXTARENA_SUITES.values())
    total = total_envs * args.textarena_seeds * 2
    done = len(completed)
    for suite_index, (suite, (all_envs, suite_dir)) in enumerate(TEXTARENA_SUITES.items()):
        envs = all_envs[: args.textarena_limit or None]
        baseline_variables = initial_slm_variables(envs)
        frozen_path = (
            FROZEN_TEXTARENA_RUN
            / suite_dir
            / "textgrad_rl_train_val_slm_final_text_variables.json"
        )
        textgrad_variables = load_variables(frozen_path)
        for env_id in envs:
            for seed_index in range(args.textarena_seeds):
                seed = args.seed + suite_index * 10_000 + seed_index
                for variant, variables in (
                    ("no_textgrad", baseline_variables),
                    ("textgrad_rl", textgrad_variables),
                ):
                    key = (suite, env_id, seed, variant)
                    if key in completed:
                        continue
                    record = run_slm_episode(
                        suite,
                        env_id,
                        variant,
                        "test",
                        seed,
                        model,
                        variables,
                    )
                    append_jsonl(path, record)
                    completed.add(key)
                    done += 1
                    print(
                        f"[TextArena {done:03d}/{total:03d}] {suite}/{env_id} "
                        f"seed={seed} {variant}: success={int(record.success)} "
                        f"reward={record.reward:.3f} turns={record.turns}",
                        flush=True,
                    )


def run_textworld(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "textworld.jsonl"
    completed = {
        (row["spec_id"], int(row["seed"]), row["variant"])
        for row in read_jsonl(path)
    }
    model = OpenAICompatibleChatModel(
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=16,
        timeout=120,
    )
    requested_problem_ids = {
        item.strip() for item in args.textworld_problems.split(",") if item.strip()
    }
    per_seed = len(requested_problem_ids) if requested_problem_ids else (args.textworld_limit or 24)
    total = per_seed * args.textworld_seeds * len(TEXTWORLD_VARIANTS)
    done = len(completed)
    for seed_index in range(args.textworld_seeds):
        game_seed = args.seed + 100_000 + seed_index * 1_000
        specs = default_specs(game_seed)
        if requested_problem_ids:
            specs = [spec for spec in specs if spec.spec_id in requested_problem_ids]
        else:
            specs = specs[: args.textworld_limit or None]
        if args.textworld_games_dir:
            game_paths = {
                spec.spec_id: Path(args.textworld_games_dir) / f"{spec.spec_id}.z8"
                for spec in specs
            }
            missing = [str(path) for path in game_paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing reused TextWorld games: {missing[:3]}")
        else:
            while True:
                try:
                    game_paths = ensure_games(
                        specs,
                        output_dir / "textworld_games" / f"seed_{game_seed}",
                    )
                    break
                except subprocess.CalledProcessError as exc:
                    print(
                        f"[TextWorld seed retry] game_seed={game_seed} failed "
                        f"({Path(exc.cmd[0]).name}, exit={exc.returncode}); trying {game_seed + 1}",
                        flush=True,
                    )
                    game_seed += 1
        for spec in specs:
            for variant in TEXTWORLD_VARIANTS:
                key = (spec.spec_id, spec.seed, variant)
                if key in completed:
                    continue
                config = VARIANT_CONFIGS[variant]
                probe = run_structured_variant(
                    variant=variant,
                    game_path=game_paths[spec.spec_id],
                    model=model,
                    use_qwen_ranker=True,
                    textgrad_rule=config["textgrad_rule"],
                    timeout=args.action_timeout,
                    max_steps=args.max_steps,
                )
                record = suite_record(
                    spec=spec,
                    record=probe,
                    model=args.model,
                    temperature=args.temperature,
                )
                append_jsonl(path, record)
                completed.add(key)
                done += 1
                print(
                    f"[TextWorld {done:03d}/{total:03d}] {spec.spec_id} seed={spec.seed} "
                    f"{variant}: success={int(record.success)} reward={record.reward:.3f} "
                    f"turns={record.turns} qwen_calls={record.qwen_calls}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/realm_multiseed_qwen7b_t07_10seed")
    parser.add_argument("--only", choices=["all", "textarena", "textworld"], default="all")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--action-timeout", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=74001)
    parser.add_argument("--textarena-seeds", type=int, default=10)
    parser.add_argument("--textworld-seeds", type=int, default=10)
    parser.add_argument("--textarena-limit", type=int, default=0)
    parser.add_argument("--textworld-limit", type=int, default=0)
    parser.add_argument("--textworld-games-dir", default="")
    parser.add_argument("--textworld-problems", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in {"all", "textarena"}:
        run_textarena(args, output_dir)
    if args.only in {"all", "textworld"}:
        run_textworld(args, output_dir)


if __name__ == "__main__":
    main()
