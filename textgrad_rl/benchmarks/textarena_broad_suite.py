"""Broad offline TextArena benchmark for TextGrad-RL policy transfer.

The policy-iteration benchmark in this repo is intentionally focused on ten
TextArena families with hand-written prompt-aware actors. This module evaluates
the resulting text policy on a wider, heterogeneous 50-environment suite. For
families outside the supported policy set, a small generic legal-action fallback
keeps the run executable and reports those environments as the fallback slice.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_multienv_compare import (
    DEFAULT_ENVS,
    MultiEnvPromptAgent,
    canonical_env_id,
)
from textgrad_rl.benchmarks.textarena_paper_suite import initial_modular_variables
from textgrad_rl.benchmarks.textarena_policy_iteration import run_policy_iteration_once
from textgrad_rl.types import TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


@dataclass(frozen=True)
class BroadTextArenaEnv:
    env_id: str
    num_players: int
    category: str


@dataclass
class BroadEpisodeRecord:
    benchmark: str
    env_id: str
    category: str
    support_group: str
    method: str
    split: str
    seed: int
    target_side: int | None
    reward: float
    success: bool
    done: bool
    turns: int
    invalid_move: bool
    repeated_actions: bool
    reason: str
    actions: list[str]
    runtime_seconds: float


BROAD_TEXTARENA_ENVS = [
    BroadTextArenaEnv("2048-v0-ultra-easy", 1, "puzzle"),
    BroadTextArenaEnv("2048-v0-mega-easy", 1, "puzzle"),
    BroadTextArenaEnv("2048-v0-super-easy", 1, "puzzle"),
    BroadTextArenaEnv("2048-v0-very-easy", 1, "puzzle"),
    BroadTextArenaEnv("2048-v0-3x3", 1, "puzzle"),
    BroadTextArenaEnv("Bandit-v0", 1, "stochastic_decision"),
    BroadTextArenaEnv("Bandit-v0-hard", 1, "stochastic_decision"),
    BroadTextArenaEnv("Blackjack-v0", 1, "stochastic_decision"),
    BroadTextArenaEnv("Blackjack-v0-long", 1, "stochastic_decision"),
    BroadTextArenaEnv("Countdown-v0", 1, "puzzle"),
    BroadTextArenaEnv("Crosswords-v0", 1, "language_puzzle"),
    BroadTextArenaEnv("Cryptarithm-v0", 1, "symbolic_puzzle"),
    BroadTextArenaEnv("FrozenLake-v0", 1, "planning"),
    BroadTextArenaEnv("FrozenLake-v0-random", 1, "planning"),
    BroadTextArenaEnv("FrozenLake-v0-hardcore", 1, "planning"),
    BroadTextArenaEnv("GuessTheNumber-v0", 1, "deduction"),
    BroadTextArenaEnv("GuessTheNumber-v0-hardcore", 1, "deduction"),
    BroadTextArenaEnv("Hangman-v0", 1, "language_puzzle"),
    BroadTextArenaEnv("Hangman-v0-hardcore", 1, "language_puzzle"),
    BroadTextArenaEnv("LightsOut-v0", 1, "puzzle"),
    BroadTextArenaEnv("LogicPuzzle-v0", 1, "symbolic_puzzle"),
    BroadTextArenaEnv("LogicPuzzle-v0-hard", 1, "symbolic_puzzle"),
    BroadTextArenaEnv("Mastermind-v0", 1, "deduction"),
    BroadTextArenaEnv("Mastermind-v0-hard", 1, "deduction"),
    BroadTextArenaEnv("Minesweeper-v0-small", 1, "puzzle"),
    BroadTextArenaEnv("PegJump-v0", 1, "puzzle"),
    BroadTextArenaEnv("RushHour-v0", 1, "planning"),
    BroadTextArenaEnv("Secretary-v0", 1, "stochastic_decision"),
    BroadTextArenaEnv("Secretary-v0-long", 1, "stochastic_decision"),
    BroadTextArenaEnv("Sudoku-v0-very-easy", 1, "symbolic_puzzle"),
    BroadTextArenaEnv("TowerOfHanoi-v0", 1, "planning"),
    BroadTextArenaEnv("TowerOfHanoi-v0-medium", 1, "planning"),
    BroadTextArenaEnv("WordLadder-v0", 1, "language_puzzle"),
    BroadTextArenaEnv("Wordle-v0", 1, "language_puzzle"),
    BroadTextArenaEnv("WordSearch-v0", 1, "language_puzzle"),
    BroadTextArenaEnv("Alquerque-v0", 2, "board_game"),
    BroadTextArenaEnv("Battleship-v0", 2, "board_game"),
    BroadTextArenaEnv("Breakthrough-v0-tiny", 2, "board_game"),
    BroadTextArenaEnv("Checkers-v0", 2, "board_game"),
    BroadTextArenaEnv("Chopsticks-v0", 2, "board_game"),
    BroadTextArenaEnv("ConnectFour-v0", 2, "board_game"),
    BroadTextArenaEnv("ConnectFour-v0-blind", 2, "board_game"),
    BroadTextArenaEnv("GameOfPureStrategy-v0", 2, "social_game"),
    BroadTextArenaEnv("GermanWhist-v0", 2, "card_game"),
    BroadTextArenaEnv("IteratedPrisonersDilemma-v0", 2, "social_game"),
    BroadTextArenaEnv("IteratedRockPaperScissors-v0", 2, "social_game"),
    BroadTextArenaEnv("KuhnPoker-v0-short", 2, "card_game"),
    BroadTextArenaEnv("Nim-v0", 2, "board_game"),
    BroadTextArenaEnv("Nim-v0-medium", 2, "board_game"),
    BroadTextArenaEnv("ReverseTicTacToe-v0", 2, "board_game"),
]


SUPPORTED_POLICY_FAMILIES = set(DEFAULT_ENVS)
METHODS = ["fixed_prompt", "textgrad_policy_iteration"]


class BroadTextArenaAgent:
    def __init__(
        self,
        env_id: str,
        variables: dict[str, TextVariable],
        *,
        player_id: int,
        target_side: int | None,
    ) -> None:
        self.env_id = env_id
        self.base_env_id = canonical_env_id(env_id)
        self.generic = GenericLegalActionAgent(env_id)
        self.supported = self.base_env_id in SUPPORTED_POLICY_FAMILIES
        self.prompt_agent = MultiEnvPromptAgent(
            env_id,
            variables,
            player_id=player_id,
            opponent=(target_side is not None and player_id != target_side),
        )

    def act(self, observation: str, previous_actions: list[str]) -> str:
        if self.supported:
            return self.prompt_agent.act(observation)
        return self.generic.act(observation, previous_actions)


class GenericLegalActionAgent:
    """Fallback policy for TextArena games outside the supported rule families."""

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.family = env_family(env_id)
        self.word_index = 0
        self.letter_index = 0
        self.secretary_seen = 0

    def act(self, observation: str, previous_actions: list[str]) -> str:
        text = observation or ""
        lower = text.lower()
        actions = parse_bracketed_actions(text)
        previous_plain = [strip_player_prefix(action) for action in previous_actions]

        if self.family == "2048":
            return first_available(actions, ["[Up]", "[Left]", "[Right]", "[Down]"])
        if self.family == "wordle":
            words = ["[arise]", "[toned]", "[clump]", "[brick]", "[swaly]", "[feast]"]
            return next_from_cycle(words, previous_plain, self.word_index)
        if self.family == "hangman":
            letters = ["[e]", "[a]", "[r]", "[i]", "[o]", "[t]", "[n]", "[s]", "[l]", "[c]"]
            return next_from_cycle(letters, previous_plain, self.letter_index)
        if self.family == "secretary":
            self.secretary_seen += 1
            value_match = re.search(r"current value is\s+([0-9.]+)", lower)
            value = float(value_match.group(1)) if value_match else 0.0
            threshold = 0.72 if "long" not in self.env_id.lower() else 0.82
            return "[accept]" if value >= threshold or self.secretary_seen >= 4 else "[continue]"
        if self.family == "iteratedprisonersdilemma":
            return "[Cooperate]" if "[Cooperate]" in actions or "cooperate" in lower else best_or_default(actions)
        if self.family == "iteratedrockpaperscissors":
            return first_available(actions, ["[Rock]", "[Paper]", "[Scissors]"])
        if self.family == "kuhnpoker":
            if "your card is: 'k'" in lower:
                return first_available(actions, ["[bet]", "[call]", "[check]", "[fold]"])
            if "your card is: 'q'" in lower:
                return first_available(actions, ["[check]", "[call]", "[bet]", "[fold]"])
            return first_available(actions, ["[check]", "[fold]", "[call]", "[bet]"])
        if self.family == "tictactoe":
            return first_available(actions, ["[4]", "[0]", "[2]", "[6]", "[8]", "[1]", "[3]", "[5]", "[7]"])
        if self.family == "chopsticks":
            return first_matching(actions, [r"\[0\s+1\]", r"\[1\s+0\]"])
        if actions:
            for action in actions:
                if action not in previous_plain:
                    return action
            return actions[0]
        if "valid moves" in lower and all(token in lower for token in ["up", "down", "left", "right"]):
            return "[Up]"
        if "accept" in lower and "continue" in lower:
            return "[continue]"
        if "cooperate" in lower and "defect" in lower:
            return "[Cooperate]"
        return "[0]"


def env_family(env_id: str) -> str:
    return env_id.split("-v0", maxsplit=1)[0].replace("_", "").replace("-", "").lower()


def parse_bracketed_actions(text: str) -> list[str]:
    actions: list[str] = []
    for match in re.finditer(r"\[[^\]\n]{1,80}\]", text):
        action = " ".join(match.group(0).split())
        inner = action[1:-1].strip()
        if not inner:
            continue
        if any(marker in inner.lower() for marker in ["game", "player", "example", "action rules"]):
            continue
        if action not in actions:
            actions.append(action)
    return actions


def strip_player_prefix(action: str) -> str:
    return re.sub(r"^p\d+:", "", action)


def first_available(actions: list[str], priority: list[str]) -> str:
    lower_to_action = {action.lower(): action for action in actions}
    for candidate in priority:
        if candidate.lower() in lower_to_action:
            return lower_to_action[candidate.lower()]
    return best_or_default(actions)


def first_matching(actions: list[str], patterns: list[str]) -> str:
    for pattern in patterns:
        for action in actions:
            if re.match(pattern, action, flags=re.IGNORECASE):
                return action
    return best_or_default(actions)


def best_or_default(actions: list[str]) -> str:
    return actions[0] if actions else "[0]"


def next_from_cycle(candidates: list[str], previous_actions: list[str], start_index: int) -> str:
    del start_index
    for candidate in candidates:
        if candidate not in previous_actions:
            return candidate
    return candidates[0]


def support_group(env_id: str) -> str:
    return "supported_policy_family" if canonical_env_id(env_id) in SUPPORTED_POLICY_FAMILIES else "generic_fallback"


def run_broad_episode(
    *,
    spec: BroadTextArenaEnv,
    method: str,
    split: str,
    seed: int,
    target_side: int | None,
    variables: dict[str, TextVariable],
    turn_budget: int,
) -> BroadEpisodeRecord:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    start = time.perf_counter()
    actions: list[str] = []
    reward = 0.0
    success = False
    invalid = False
    reason = ""
    done = False
    turns = 0
    env = None
    try:
        env = ta.make(spec.env_id)
        env.reset(num_players=spec.num_players, seed=seed)
        agents = {
            pid: BroadTextArenaAgent(spec.env_id, variables, player_id=pid, target_side=target_side)
            for pid in range(spec.num_players)
        }
        while not done and turns < turn_budget:
            pid, observation = env.get_observation()
            if not isinstance(observation, str):
                observation = "\n".join(str(item) for item in observation)
            action = agents[pid].act(observation, actions)
            action = " ".join(action.strip().split()) or "[0]"
            actions.append(f"p{pid}:{action}")
            done, _ = env.step(action)
            turns += 1
        if not done:
            reason = "turn budget exhausted"
        rewards, game_info = env.close()
        target = 0 if target_side is None else target_side
        if isinstance(rewards, dict):
            reward = float(rewards.get(target, 0.0))
        target_info = game_info.get(target, {}) if isinstance(game_info, dict) else {}
        invalid = bool(target_info.get("invalid_move", False))
        reason = reason or str(target_info.get("reason", ""))
        success = reward >= 1.0 if spec.num_players == 1 else reward > 0
    except Exception as exc:
        invalid = True
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None and not done:
            try:
                env.close()
            except Exception:
                pass

    plain_actions = [strip_player_prefix(action) for action in actions]
    return BroadEpisodeRecord(
        benchmark="textarena_broad",
        env_id=spec.env_id,
        category=spec.category,
        support_group=support_group(spec.env_id),
        method=method,
        split=split,
        seed=seed,
        target_side=target_side,
        reward=reward,
        success=success,
        done=done,
        turns=turns,
        invalid_move=invalid,
        repeated_actions=len(plain_actions) != len(set(plain_actions)),
        reason=reason,
        actions=actions,
        runtime_seconds=time.perf_counter() - start,
    )


def run_broad_records(
    *,
    specs: list[BroadTextArenaEnv],
    method: str,
    split: str,
    seeds_per_env: int,
    seed: int,
    variables: dict[str, TextVariable],
    turn_budget: int,
    output_jsonl: Path,
) -> list[BroadEpisodeRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[BroadEpisodeRecord] = []
    for spec in specs:
        target_sides = [None] if spec.num_players == 1 else list(range(spec.num_players))
        for target_side in target_sides:
            for index in range(seeds_per_env):
                record = run_broad_episode(
                    spec=spec,
                    method=method,
                    split=split,
                    seed=seed + index,
                    target_side=target_side,
                    variables=variables,
                    turn_budget=turn_budget,
                )
                records.append(record)
                append_jsonl(output_jsonl, record)
    return records


def summarize_records(method: str, records: list[BroadEpisodeRecord], accepted_updates: int) -> dict[str, Any]:
    return {
        "method": method,
        "envs": len({record.env_id for record in records}),
        "episodes": len(records),
        "average_reward": mean([record.reward for record in records]),
        "success_rate": mean([float(record.success) for record in records]),
        "invalid_move_rate": mean([float(record.invalid_move) for record in records]),
        "repeated_action_rate": mean([float(record.repeated_actions) for record in records]),
        "average_turns": mean([record.turns for record in records]),
        "supported_envs": len(
            {record.env_id for record in records if record.support_group == "supported_policy_family"}
        ),
        "fallback_envs": len({record.env_id for record in records if record.support_group == "generic_fallback"}),
        "accepted_updates": accepted_updates,
    }


def summarize_groups(records: list[BroadEpisodeRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({record.method for record in records}):
        method_records = [record for record in records if record.method == method]
        for key_name in ["support_group", "category"]:
            for key_value in sorted({getattr(record, key_name) for record in method_records}):
                group = [record for record in method_records if getattr(record, key_name) == key_value]
                rows.append(
                    {
                        "method": method,
                        "slice": key_name,
                        "value": key_value,
                        "envs": len({record.env_id for record in group}),
                        "episodes": len(group),
                        "average_reward": mean([record.reward for record in group]),
                        "success_rate": mean([float(record.success) for record in group]),
                        "invalid_move_rate": mean([float(record.invalid_move) for record in group]),
                        "repeated_action_rate": mean([float(record.repeated_actions) for record in group]),
                        "average_turns": mean([record.turns for record in group]),
                    }
                )
    return rows


def summarize_per_env(records: list[BroadEpisodeRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({record.method for record in records}):
        method_records = [record for record in records if record.method == method]
        for env_id in sorted({record.env_id for record in method_records}):
            group = [record for record in method_records if record.env_id == env_id]
            rows.append(
                {
                    "method": method,
                    "env_id": env_id,
                    "category": group[0].category,
                    "support_group": group[0].support_group,
                    "episodes": len(group),
                    "average_reward": mean([record.reward for record in group]),
                    "success_rate": mean([float(record.success) for record in group]),
                    "invalid_move_rate": mean([float(record.invalid_move) for record in group]),
                    "repeated_action_rate": mean([float(record.repeated_actions) for record in group]),
                    "average_turns": mean([record.turns for record in group]),
                }
            )
    return rows


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary_markdown(
    path: Path,
    *,
    specs: list[BroadTextArenaEnv],
    summary_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# TextArena Broad 50-Environment Suite",
        "",
        f"Environments: {len(specs)}",
        f"Test seeds: {args.test_seeds}",
        f"Turn budget: {args.turn_budget}",
        f"Supported policy-family envs: {sum(1 for spec in specs if support_group(spec.env_id) == 'supported_policy_family')}",
        f"Generic fallback envs: {sum(1 for spec in specs if support_group(spec.env_id) == 'generic_fallback')}",
        "",
        "## Overall Results",
        "",
        "| Method | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {envs} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_move_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} | {accepted_updates} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Supported vs Fallback",
            "",
            "| Method | Slice | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [row for row in group_rows if row["slice"] == "support_group"]:
        lines.append(
            "| {method} | {value} | {envs} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_move_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Category Results",
            "",
            "| Method | Category | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [row for row in group_rows if row["slice"] == "category"]:
        lines.append(
            "| {method} | {value} | {envs} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_move_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def variables_for_method(
    method: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, TextVariable], int]:
    if method == "fixed_prompt":
        return initial_modular_variables(), 0
    if method == "textgrad_policy_iteration":
        rows, _test_records, variables = run_policy_iteration_once(
            env_ids=list(DEFAULT_ENVS),
            repetition=0,
            train_seeds=args.train_seeds,
            val_seeds=args.val_seeds,
            test_seeds=args.policy_training_test_seeds,
            seed=args.seed,
            output_dir=output_dir / "policy_training",
            min_mean_delta=args.min_mean_delta,
            max_ci_low_regression=args.max_ci_low_regression,
        )
        accepted = max((int(row.get("accepted_count", 0)) for row in rows), default=0)
        return variables, accepted
    raise ValueError(f"Unknown method: {method}")


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods or list(METHODS)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_methods(args.methods)
    specs = list(BROAD_TEXTARENA_ENVS)
    write_json(
        output_dir / "config.json",
        {
            "benchmark": "textarena_broad",
            "envs": specs,
            "methods": methods,
            "train_envs": list(DEFAULT_ENVS),
            "train_seeds": args.train_seeds,
            "val_seeds": args.val_seeds,
            "test_seeds": args.test_seeds,
            "policy_training_test_seeds": args.policy_training_test_seeds,
            "seed": args.seed,
            "turn_budget": args.turn_budget,
        },
    )
    write_json(output_dir / "environment_info.json", environment_info())
    all_records: list[BroadEpisodeRecord] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        variables, accepted_updates = variables_for_method(method, args=args, output_dir=output_dir)
        write_json(output_dir / f"{method}_text_variables.json", variables)
        records = run_broad_records(
            specs=specs,
            method=method,
            split="test",
            seeds_per_env=args.test_seeds,
            seed=args.seed + 100_000,
            variables=variables,
            turn_budget=args.turn_budget,
            output_jsonl=output_dir / f"{method}_episodes.jsonl",
        )
        all_records.extend(records)
        summary_rows.append(summarize_records(method, records, accepted_updates))
    group_rows = summarize_groups(all_records)
    per_env_rows = summarize_per_env(all_records)
    write_json(output_dir / "summary.json", summary_rows)
    write_json(output_dir / "slice_summary.json", group_rows)
    write_json(output_dir / "per_env_metrics.json", per_env_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "slice_summary.csv", group_rows)
    write_csv(output_dir / "per_env_metrics.csv", per_env_rows)
    write_summary_markdown(output_dir / "summary.md", specs=specs, summary_rows=summary_rows, group_rows=group_rows, args=args)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a broad 50-environment offline TextArena benchmark.")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--policy-training-test-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--turn-budget", type=int, default=80)
    parser.add_argument("--min-mean-delta", type=float, default=0.001)
    parser.add_argument("--max-ci-low-regression", type=float, default=0.0)
    parser.add_argument("--output-dir", default="runs/textarena_broad_50")
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    print(f"TextArena broad suite artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
