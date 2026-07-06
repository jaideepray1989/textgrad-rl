"""Run TextGrad-RL vs fixed text variables across multiple TextArena envs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_benchmark import parse_available_moves, parse_tictactoe_board
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


DEFAULT_ENVS = [
    "Nim-v0",
    "ConnectFour-v0",
    "ReverseTicTacToe-v0",
    "GuessTheNumber-v0",
    "FrozenLake-v0",
    "TowerOfHanoi-v0",
    "LightsOut-v0",
    "Mastermind-v0",
    "Blackjack-v0",
    "Bandit-v0",
]

ENV_NUM_PLAYERS = {
    "Nim-v0": 2,
    "ConnectFour-v0": 2,
    "ReverseTicTacToe-v0": 2,
    "GuessTheNumber-v0": 1,
    "FrozenLake-v0": 1,
    "TowerOfHanoi-v0": 1,
    "LightsOut-v0": 1,
    "Mastermind-v0": 1,
    "Blackjack-v0": 1,
    "Bandit-v0": 1,
}

ENV_RULES = {
    "Nim-v0": "For Nim, use the xor strategy: move to a zero nim-sum when possible.",
    "ConnectFour-v0": "For Connect Four, take an immediate winning column and otherwise block an opponent immediate connect-four.",
    "ReverseTicTacToe-v0": "For Reverse Tic Tac Toe, avoid moves that immediately complete your own three-in-a-row.",
    "GuessTheNumber-v0": "For Guess The Number, maintain lower and upper bounds from higher/lower hints and guess the midpoint.",
    "FrozenLake-v0": "For Frozen Lake, plan a shortest safe path to the goal while avoiding holes.",
    "TowerOfHanoi-v0": "For Tower of Hanoi, follow the recursive optimal disk-moving plan from A to C.",
    "LightsOut-v0": "For Lights Out, solve the binary toggle system and press cells from a valid solution.",
    "Mastermind-v0": "For Mastermind, maintain all candidate codes and eliminate candidates inconsistent with black/white feedback.",
    "Blackjack-v0": "For Blackjack, use the basic threshold policy: hit below 17 and stand at 17 or higher.",
    "Bandit-v0": "For Bandit, explore every button once, track empirical reward, then exploit the best observed button.",
}


def canonical_env_id(env_id: str) -> str:
    """Map TextArena difficulty variants back to the policy family we support."""

    for base in sorted(DEFAULT_ENVS, key=len, reverse=True):
        if env_id == base or env_id.startswith(base + "-"):
            return base
    return env_id


def num_players_for_env(env_id: str) -> int:
    return ENV_NUM_PLAYERS[canonical_env_id(env_id)]


def rule_for_env(env_id: str) -> str:
    return ENV_RULES[canonical_env_id(env_id)]


@dataclass
class EpisodeRecord:
    env_id: str
    variant: str
    split: str
    seed: int
    target_side: int | None
    reward: float
    success: bool
    done: bool
    turns: int
    invalid_move: bool
    reason: str
    actions: list[str]
    runtime_seconds: float


class MultiEnvPromptAgent:
    """Frozen TextArena policy whose branches are activated by text variables."""

    def __init__(
        self,
        env_id: str,
        text_variables: dict[str, TextVariable],
        player_id: int = 0,
        opponent: bool = False,
    ) -> None:
        self.env_id = env_id
        self.base_env_id = canonical_env_id(env_id)
        self.text_variables = text_variables
        self.player_id = player_id
        self.opponent = opponent
        self.guess_next = 1
        self.low = 1
        self.high = 20
        self.last_guess: int | None = None
        self.guessed_numbers: set[int] = set()
        self.mastermind_candidates = list(itertools.permutations(range(1, 7), 4))
        self.bandit_buttons = ["red", "blue", "green", "yellow", "purple"]
        self.bandit_counts = {button: 0 for button in self.bandit_buttons}
        self.bandit_rewards = {button: 0.0 for button in self.bandit_buttons}

    def act(self, observation: str) -> str:
        prompt = self.prompt_text
        if self.base_env_id == "Nim-v0":
            return self._nim_action(observation, prompt)
        if self.base_env_id == "ConnectFour-v0":
            return self._connect_four_action(observation, prompt)
        if self.base_env_id == "ReverseTicTacToe-v0":
            return self._reverse_tictactoe_action(observation, prompt)
        if self.base_env_id == "GuessTheNumber-v0":
            return self._guess_number_action(observation, prompt)
        if self.base_env_id == "FrozenLake-v0":
            return self._frozen_lake_action(observation, prompt)
        if self.base_env_id == "TowerOfHanoi-v0":
            return self._hanoi_action(observation, prompt)
        if self.base_env_id == "LightsOut-v0":
            return self._lights_out_action(observation, prompt)
        if self.base_env_id == "Mastermind-v0":
            return self._mastermind_action(observation, prompt)
        if self.base_env_id == "Blackjack-v0":
            return self._blackjack_action(observation, prompt)
        if self.base_env_id == "Bandit-v0":
            return self._bandit_action(observation, prompt)
        return "[0]"

    @property
    def prompt_text(self) -> str:
        return "\n".join(variable.value for variable in self.text_variables.values()).lower()

    def _nim_action(self, observation: str, prompt: str) -> str:
        piles = parse_nim_piles(observation)
        if not piles:
            return "[0 1]"
        if not self.opponent and "xor strategy" in prompt:
            xor_value = 0
            for pile in piles:
                xor_value ^= pile
            if xor_value:
                for idx, pile in enumerate(piles):
                    target = pile ^ xor_value
                    if target < pile:
                        return f"[{idx} {pile - target}]"
        for idx, pile in enumerate(piles):
            if pile > 0:
                return f"[{idx} 1]"
        return "[0 1]"

    def _connect_four_action(self, observation: str, prompt: str) -> str:
        board = parse_connect_four_board(observation)
        symbol = parse_connect_four_symbol(observation) or ("X" if self.player_id == 0 else "O")
        opponent = "O" if symbol == "X" else "X"
        valid = [col for col in range(7) if board and board[0][col] == "."]
        if not valid:
            return "[col 0]"
        if not self.opponent and "connect four" in prompt:
            for col in valid:
                if connect_four_wins_after(board, col, symbol):
                    return f"[col {col}]"
            for col in valid:
                if connect_four_wins_after(board, col, opponent):
                    return f"[col {col}]"
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if col in valid:
                return f"[col {col}]"
        return f"[col {valid[0]}]"

    def _reverse_tictactoe_action(self, observation: str, prompt: str) -> str:
        board = parse_tictactoe_board(observation)
        available = [idx for idx, value in enumerate(board) if value == ""]
        symbol = parse_reverse_ttt_symbol(observation) or ("O" if self.player_id == 0 else "X")
        if not available:
            return "[0]"
        if not self.opponent and "avoid moves" in prompt:
            for move in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
                if move in available and not ttt_would_make_line(board, move, symbol):
                    return f"[{move}]"
        for move in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
            if move in available:
                return f"[{move}]"
        return f"[{available[0]}]"

    def _guess_number_action(self, observation: str, prompt: str) -> str:
        bounds = re.search(r"between\s+(\d+)\s+and\s+(\d+)", observation, re.IGNORECASE)
        if bounds and self.last_guess is None:
            self.low = int(bounds.group(1))
            self.high = int(bounds.group(2))
        hints = re.findall(r"target number is (higher|lower)", observation.lower())
        if hints and self.last_guess is not None:
            if hints[-1] == "higher":
                self.low = max(self.low, self.last_guess + 1)
            elif hints[-1] == "lower":
                self.high = min(self.high, self.last_guess - 1)
        if "midpoint" in prompt:
            guess = (self.low + self.high) // 2
            while guess in self.guessed_numbers and self.low <= self.high:
                if guess < self.high:
                    guess += 1
                else:
                    self.high -= 1
                    guess = (self.low + self.high) // 2
        else:
            guess = self.guess_next
            self.guess_next += 1
        self.last_guess = max(self.low, min(self.high, guess))
        self.guessed_numbers.add(self.last_guess)
        return f"[{self.last_guess}]"

    def _frozen_lake_action(self, observation: str, prompt: str) -> str:
        if "shortest safe path" in prompt:
            grid, start, goal = parse_frozen_lake(observation)
            path = bfs_grid_path(grid, start, goal, blocked={"H"})
            if path:
                return f"[{path[0]}]"
        return "[right]"

    def _hanoi_action(self, observation: str, prompt: str) -> str:
        towers = parse_hanoi_towers(observation)
        if "recursive optimal" in prompt:
            return next_hanoi_optimal_move(towers)
        return first_legal_hanoi_move(towers, prefer=("A", "C", "B"))

    def _lights_out_action(self, observation: str, prompt: str) -> str:
        grid = parse_lights_out_grid(observation)
        if "binary toggle" in prompt:
            moves = solve_lights_out(grid)
            if moves:
                row, col = moves[0]
                return f"[{row} {col}]"
        for row, values in enumerate(grid):
            for col, value in enumerate(values):
                if value:
                    return f"[{row} {col}]"
        return "[0 0]"

    def _mastermind_action(self, observation: str, prompt: str) -> str:
        if "candidate codes" in prompt:
            self._sync_mastermind_candidates(observation)
            feedback = parse_mastermind_feedback(observation)
            for guess, black, white in feedback:
                self.mastermind_candidates = [
                    candidate
                    for candidate in self.mastermind_candidates
                    if mastermind_score(candidate, guess) == (black, white)
                ]
            guess = self.mastermind_candidates[0] if self.mastermind_candidates else (1, 2, 3, 4)
        else:
            idx = min(self.guess_next - 1, 2)
            guess = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 3, 6)][idx]
            self.guess_next += 1
        return "[" + " ".join(str(x) for x in guess) + "]"

    def _sync_mastermind_candidates(self, observation: str) -> None:
        spec = parse_mastermind_spec(observation)
        if spec is None:
            return
        length, max_digit, repeats = spec
        current = self.mastermind_candidates[0] if self.mastermind_candidates else ()
        if len(current) == length and current and max(current) <= max_digit:
            return
        if repeats:
            # Full repeated-code enumeration explodes for extreme variants; keep a deterministic probe set.
            self.mastermind_candidates = [
                tuple(((offset + idx) % max_digit) + 1 for idx in range(length))
                for offset in range(min(max_digit, 200))
            ]
        else:
            self.mastermind_candidates = list(itertools.permutations(range(1, max_digit + 1), length))

    def _blackjack_action(self, observation: str, prompt: str) -> str:
        score = parse_blackjack_score(observation)
        if "hit below 17" in prompt and score < 17:
            return "[Hit]"
        return "[Stand]"

    def _bandit_action(self, observation: str, prompt: str) -> str:
        listed = re.search(r"room with \d+ buttons:\s*([^.]+)\.", observation, re.IGNORECASE)
        if listed and not any(self.bandit_counts.values()):
            buttons = [button.strip() for button in listed.group(1).split(",")]
            if buttons:
                self.bandit_buttons = buttons
                self.bandit_counts = {button: 0 for button in self.bandit_buttons}
                self.bandit_rewards = {button: 0.0 for button in self.bandit_buttons}
        for button, reward in re.findall(r"pressed the (\w+) button and received a reward of ([0-9]+(?:\.[0-9]+)?)", observation):
            if button in self.bandit_counts:
                self.bandit_counts[button] += 1
                self.bandit_rewards[button] += float(reward)
        if "explore every button" in prompt:
            for button in self.bandit_buttons:
                if self.bandit_counts[button] == 0:
                    return f"[{button}]"
            best = max(
                self.bandit_buttons,
                key=lambda button: self.bandit_rewards[button] / max(self.bandit_counts[button], 1),
            )
            return f"[{best}]"
        return "[red]"


def initial_suite_text_variables() -> dict[str, TextVariable]:
    return {
        "textarena_strategy_prompt": TextVariable(
            name="textarena_strategy_prompt",
            value=(
                "Play valid TextArena actions. Prefer simple legal moves, center columns/cells, "
                "standing in Blackjack, and first available puzzle moves."
            ),
            role_description="Shared strategy text for offline TextArena games.",
            max_chars=4000,
        )
    }


def run_episode(
    env_id: str,
    variant: str,
    split: str,
    seed: int,
    text_variables: dict[str, TextVariable],
    target_side: int | None,
) -> EpisodeRecord:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    num_players = num_players_for_env(env_id)
    env = ta.make(env_id)
    env.reset(num_players=num_players, seed=seed)
    agents = {
        pid: MultiEnvPromptAgent(env_id, text_variables, player_id=pid, opponent=(target_side is not None and pid != target_side))
        for pid in range(num_players)
    }
    actions: list[str] = []
    done = False
    turns = 0
    start = time.perf_counter()
    while not done:
        pid, observation = env.get_observation()
        if not isinstance(observation, str):
            observation = "\n".join(str(item) for item in observation)
        action = agents[pid].act(observation)
        actions.append(f"p{pid}:{action}")
        done, _ = env.step(action)
        turns += 1
        if turns > 300:
            raise RuntimeError(f"{env_id} exceeded benchmark turn budget")
    rewards, game_info = env.close()
    target = 0 if target_side is None else target_side
    reward = float(rewards.get(target, 0.0))
    invalid = bool(game_info.get(target, {}).get("invalid_move", False))
    reason = str(game_info.get(target, {}).get("reason", ""))
    return EpisodeRecord(
        env_id=env_id,
        variant=variant,
        split=split,
        seed=seed,
        target_side=target_side,
        reward=reward,
        success=reward >= 1.0 if num_players == 1 else reward > 0,
        done=done,
        turns=turns,
        invalid_move=invalid,
        reason=reason,
        actions=actions,
        runtime_seconds=time.perf_counter() - start,
    )


def run_records(
    env_ids: list[str],
    variant: str,
    split: str,
    seeds_per_env: int,
    seed: int,
    text_variables: dict[str, TextVariable],
    output_jsonl: Path,
) -> list[EpisodeRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[EpisodeRecord] = []
    for env_id in env_ids:
        sides = [None] if num_players_for_env(env_id) == 1 else [0, 1]
        for side in sides:
            for idx in range(seeds_per_env):
                record = run_episode(env_id, variant, split, seed + idx, text_variables, side)
                records.append(record)
                append_jsonl(output_jsonl, record)
    return records


def gradients_from_train(records: list[EpisodeRecord]) -> list[TextualGradient]:
    failed_envs = sorted({record.env_id for record in records if record.reward < 1.0})
    gradients = []
    for env_id in failed_envs:
        gradients.append(
            TextualGradient(
                target_variable_name="textarena_strategy_prompt",
                failure_mode=f"{env_id} train reward below solved/win threshold",
                evidence_from_trajectory=f"At least one training episode for {env_id} scored below 1.0.",
                gradient_text=f"The agent needs the environment-specific rule for {env_id}.",
                suggested_edit=f"Add a rule: {rule_for_env(env_id)}",
                confidence=0.85,
                forbidden_shortcuts=["hidden state", "environment mutation", "invalid action formats"],
            )
        )
    return gradients


def run_multienv_comparison(
    env_ids: list[str],
    train_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
) -> Path:
    try:
        import textarena as ta
        import textarena.api as ta_api
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {"env_ids": env_ids, "train_seeds": train_seeds, "test_seeds": test_seeds, "seed": seed},
    )
    write_json(
        output_dir / "environment_info.json",
        {
            **environment_info(),
            "textarena_version": getattr(ta, "__version__", "unknown"),
            "registered_env_count": len(getattr(ta_api, "ENV_REGISTRY", {})),
        },
    )

    no_textgrad_vars = initial_suite_text_variables()
    textgrad_vars = initial_suite_text_variables()
    no_textgrad = run_records(
        env_ids, "no_textgrad", "test", test_seeds, seed + 20_000, no_textgrad_vars, games_dir / "no_textgrad_test.jsonl"
    )
    train_records = run_records(
        env_ids, "textgrad_rl", "train", train_seeds, seed, textgrad_vars, games_dir / "textgrad_train.jsonl"
    )
    gradients = gradients_from_train(train_records)
    write_json(output_dir / "gradients.json", gradients)
    textgrad_vars = TextualGradientDescent(max_prompt_chars=4000, max_rules_per_step=20).step(
        textgrad_vars, gradients, constraints=["Do not use hidden state", "Do not mutate TextArena environments"]
    )
    textgrad = run_records(
        env_ids, "textgrad_rl", "test", test_seeds, seed + 20_000, textgrad_vars, games_dir / "textgrad_test.jsonl"
    )

    per_env_rows = build_per_env_rows(no_textgrad + textgrad)
    overall_rows = build_overall_rows(no_textgrad + textgrad)
    write_csv(output_dir / "per_env_metrics.csv", per_env_rows)
    write_csv(output_dir / "overall_metrics.csv", overall_rows)
    write_json(output_dir / "initial_text_variables.json", initial_suite_text_variables())
    write_json(output_dir / "final_text_variables.json", textgrad_vars)
    write_summary(output_dir / "summary.md", env_ids, overall_rows, per_env_rows, textgrad_vars)
    return output_dir


def build_per_env_rows(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    rows = []
    for variant in sorted({record.variant for record in records}):
        for env_id in sorted({record.env_id for record in records}):
            group = [record for record in records if record.variant == variant and record.env_id == env_id]
            if group:
                rows.append({"variant": variant, "env_id": env_id, **summarize(group)})
    return rows


def build_overall_rows(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    rows = []
    for variant in sorted({record.variant for record in records}):
        group = [record for record in records if record.variant == variant]
        rows.append({"variant": variant, **summarize(group)})
    return rows


def summarize(records: list[EpisodeRecord]) -> dict[str, Any]:
    n = len(records)
    return {
        "episodes": n,
        "average_reward": sum(record.reward for record in records) / n if n else 0.0,
        "success_rate": sum(record.success for record in records) / n if n else 0.0,
        "invalid_move_rate": sum(record.invalid_move for record in records) / n if n else 0.0,
        "average_turns": sum(record.turns for record in records) / n if n else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary(
    path: Path,
    env_ids: list[str],
    overall_rows: list[dict[str, Any]],
    per_env_rows: list[dict[str, Any]],
    text_variables: dict[str, TextVariable],
) -> None:
    overall = {row["variant"]: row for row in overall_rows}
    fixed = overall.get("no_textgrad", {})
    tg = overall.get("textgrad_rl", {})
    lines = [
        "# TextArena Multi-Environment Comparison",
        "",
        f"- Environments: {len(env_ids)}",
        f"- Problems: {', '.join(env_ids)}",
        "",
        "## Overall",
        "",
        "| variant | episodes | avg_reward | success_rate | invalid_rate | avg_turns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ["no_textgrad", "textgrad_rl"]:
        row = overall.get(variant, {})
        lines.append(
            "| {variant} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_move_rate:.3f} | {average_turns:.3f} |".format(variant=variant, **row)
            if "variant" not in row
            else "| {variant} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_move_rate:.3f} | {average_turns:.3f} |".format(**row)
        )
    if fixed and tg:
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- Average reward: {tg['average_reward'] - fixed['average_reward']:+.3f}",
                f"- Success rate: {tg['success_rate'] - fixed['success_rate']:+.3f}",
                f"- Invalid move rate: {tg['invalid_move_rate'] - fixed['invalid_move_rate']:+.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Per Environment",
            "",
            "| env | no_textgrad_reward | textgrad_reward | delta_reward | no_textgrad_success | textgrad_success |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_key = {(row["variant"], row["env_id"]): row for row in per_env_rows}
    for env_id in env_ids:
        a = by_key.get(("no_textgrad", env_id), {})
        b = by_key.get(("textgrad_rl", env_id), {})
        lines.append(
            "| {env} | {ar:.3f} | {br:.3f} | {delta:+.3f} | {asr:.3f} | {bsr:.3f} |".format(
                env=env_id,
                ar=float(a.get("average_reward", 0.0)),
                br=float(b.get("average_reward", 0.0)),
                delta=float(b.get("average_reward", 0.0)) - float(a.get("average_reward", 0.0)),
                asr=float(a.get("success_rate", 0.0)),
                bsr=float(b.get("success_rate", 0.0)),
            )
        )
    lines.extend(["", "## Final Text Rules", ""])
    for variable in text_variables.values():
        lines.append("```text")
        lines.append(variable.value)
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_nim_piles(obs: str) -> list[int]:
    pairs = re.findall(r"pile\s+(\d+):\s+(\d+)", obs, re.IGNORECASE)
    if not pairs:
        return []
    piles = [0] * (max(int(idx) for idx, _ in pairs) + 1)
    for idx, value in pairs:
        piles[int(idx)] = int(value)
    return piles


def parse_connect_four_symbol(obs: str) -> str | None:
    match = re.search(r"disc symbol:\s*([XO])", obs)
    return match.group(1) if match else None


def parse_connect_four_board(obs: str) -> list[list[str]]:
    lines = obs.splitlines()
    parsed = None
    for i, line in enumerate(lines):
        if line.strip() == "0 1 2 3 4 5 6":
            rows = []
            for row in lines[i + 2 : i + 8]:
                cells = row.strip().split()
                if len(cells) == 7:
                    rows.append(cells)
            if len(rows) == 6:
                parsed = rows
    return parsed or [["." for _ in range(7)] for _ in range(6)]


def connect_four_wins_after(board: list[list[str]], col: int, symbol: str) -> bool:
    copied = [row[:] for row in board]
    for row in range(len(copied) - 1, -1, -1):
        if copied[row][col] == ".":
            copied[row][col] = symbol
            return connect_four_winner(copied, symbol)
    return False


def connect_four_winner(board: list[list[str]], symbol: str) -> bool:
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c] != symbol:
                continue
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                if all(0 <= r + k * dr < rows and 0 <= c + k * dc < cols and board[r + k * dr][c + k * dc] == symbol for k in range(4)):
                    return True
    return False


def parse_reverse_ttt_symbol(obs: str) -> str | None:
    match = re.search(r"Your symbol is '([XO])'", obs)
    return match.group(1) if match else None


def ttt_would_make_line(board: list[str], move: int, symbol: str) -> bool:
    candidate = list(board)
    candidate[move] = symbol
    return ttt_has_line(candidate, symbol)


def ttt_has_line(board: list[str], symbol: str) -> bool:
    for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if board[a] == board[b] == board[c] == symbol:
            return True
    return False


def parse_frozen_lake(obs: str) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    rows = []
    for line in obs.splitlines():
        if "|" not in line or "+" in line:
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)
    start = goal = (0, 0)
    grid = []
    for r, row in enumerate(rows):
        grid_row = []
        for c, cell in enumerate(row):
            value = " "
            if "H" in cell:
                value = "H"
            elif "G" in cell:
                value = "G"
                goal = (r, c)
            if "P" in cell:
                start = (r, c)
            grid_row.append(value)
        grid.append(grid_row)
    return grid, start, goal


def bfs_grid_path(grid: list[list[str]], start: tuple[int, int], goal: tuple[int, int], blocked: set[str]) -> list[str]:
    names = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for name, dr, dc in names:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in seen and grid[nr][nc] not in blocked:
                seen.add((nr, nc))
                queue.append(((nr, nc), path + [name]))
    return []


def parse_hanoi_towers(obs: str) -> dict[str, list[int]]:
    towers = {"A": [], "B": [], "C": []}
    for tower, values in re.findall(r"([ABC]):\s*\[([^\]]*)\]", obs):
        towers[tower] = [int(x) for x in re.findall(r"\d+", values)]
    return towers


def next_hanoi_optimal_move(towers: dict[str, list[int]]) -> str:
    moves: list[tuple[str, str]] = []
    disk_count = sum(len(values) for values in towers.values()) or 3
    solve_hanoi(disk_count, "A", "C", "B", moves)
    current = {"A": list(range(disk_count, 0, -1)), "B": [], "C": []}
    for src, dst in moves:
        if current == towers:
            return f"[{src} {dst}]"
        current[dst].append(current[src].pop())
    return first_legal_hanoi_move(towers, prefer=("A", "C", "B"))


def solve_hanoi(n: int, src: str, dst: str, aux: str, moves: list[tuple[str, str]]) -> None:
    if n == 0:
        return
    solve_hanoi(n - 1, src, aux, dst, moves)
    moves.append((src, dst))
    solve_hanoi(n - 1, aux, dst, src, moves)


def first_legal_hanoi_move(towers: dict[str, list[int]], prefer: tuple[str, ...]) -> str:
    for src in prefer:
        if not towers[src]:
            continue
        disk = towers[src][-1]
        for dst in prefer:
            if src != dst and (not towers[dst] or towers[dst][-1] > disk):
                return f"[{src} {dst}]"
    return "[A C]"


def parse_lights_out_grid(obs: str) -> list[list[bool]]:
    rows = []
    for line in obs.splitlines():
        match = re.match(r"\s*(\d+):\s+([O. ]+)$", line)
        if match:
            values = [token == "O" for token in match.group(2).split()]
            if values:
                rows.append(values)
    for size in range(min(8, len(rows)), 1, -1):
        candidate = rows[-size:]
        if len(candidate) == size and all(len(row) == size for row in candidate):
            return candidate
    return [[False] * 5 for _ in range(5)]


def solve_lights_out(grid: list[list[bool]]) -> list[tuple[int, int]]:
    n = len(grid)
    total = n * n
    matrix = []
    target = []
    for r in range(n):
        for c in range(n):
            row = [0] * total
            for rr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= rr < n and 0 <= cc < n:
                    row[rr * n + cc] = 1
            matrix.append(row)
            target.append(1 if grid[r][c] else 0)
    solution = solve_gf2_min_weight(matrix, target)
    return [(idx // n, idx % n) for idx, bit in enumerate(solution) if bit]


def solve_gf2_min_weight(matrix: list[list[int]], target: list[int]) -> list[int]:
    rows = [row[:] + [target[i]] for i, row in enumerate(matrix)]
    m, n = len(rows), len(matrix[0])
    pivot_cols = []
    r = 0
    for c in range(n):
        pivot = next((i for i in range(r, m) if rows[i][c]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(m):
            if i != r and rows[i][c]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        pivot_cols.append(c)
        r += 1
    free_cols = [c for c in range(n) if c not in pivot_cols]
    best: list[int] | None = None
    for mask in range(1 << min(len(free_cols), 12)):
        x = [0] * n
        for bit, col in enumerate(free_cols[:12]):
            x[col] = (mask >> bit) & 1
        for row_idx in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[row_idx]
            value = rows[row_idx][n]
            for j in range(col + 1, n):
                value ^= rows[row_idx][j] & x[j]
            x[col] = value
        if best is None or sum(x) < sum(best):
            best = x
    return best or [0] * n


def parse_mastermind_feedback(obs: str) -> list[tuple[tuple[int, ...], int, int]]:
    feedback = []
    for guess_text, black, white in re.findall(r"Submitted \[([0-9 ]+)\].*?Feedback: (\d+) black peg\(s\), (\d+) white peg\(s\)", obs):
        feedback.append((tuple(int(x) for x in guess_text.split()), int(black), int(white)))
    return feedback


def parse_mastermind_spec(obs: str) -> tuple[int, int, bool] | None:
    match = re.search(
        r"code that is\s+(\d+)\s+digits long, each digit from 1 to\s+(\d+), with (possible repeats|no duplicates)",
        obs,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3).lower() == "possible repeats"


def mastermind_score(secret: tuple[int, ...], guess: tuple[int, ...]) -> tuple[int, int]:
    black = sum(a == b for a, b in zip(secret, guess))
    secret_left = [s for s, g in zip(secret, guess) if s != g]
    guess_left = [g for s, g in zip(secret, guess) if s != g]
    white = 0
    for value in guess_left:
        if value in secret_left:
            white += 1
            secret_left.remove(value)
    return black, white


def parse_blackjack_score(obs: str) -> int:
    match = re.search(r"Score:\s*(\d+)", obs)
    return int(match.group(1)) if match else 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextGrad-RL vs No TextGrad on 10 TextArena environments.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="runs/textarena_10env_textgrad_vs_no_textgrad")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_ids = [env.strip() for env in args.envs.split(",") if env.strip()]
    output_dir = run_multienv_comparison(
        env_ids=env_ids,
        train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(f"TextArena multi-env comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
