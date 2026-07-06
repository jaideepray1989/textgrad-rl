"""Offline TextArena benchmark runner.

The default benchmark is a full deterministic round-robin on TextArena's
TicTacToe-v0 environment. It avoids remote model APIs while exercising the real
TextArena package API.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


class TextArenaAgent(Protocol):
    name: str

    def __call__(self, observation: str) -> str: ...


@dataclass
class GameResult:
    env_id: str
    seed: int
    player0_agent: str
    player1_agent: str
    rewards: dict[int, float]
    winner: int | None
    draw: bool
    turns: int
    invalid_moves: dict[int, bool]
    reason: str
    actions: list[dict[str, Any]]
    runtime_seconds: float


class FirstAvailableAgent:
    name = "first_available"

    def __call__(self, observation: str) -> str:
        available = parse_available_moves(observation)
        return f"[{available[0]}]" if available else "[0]"


class CenterFirstAgent:
    name = "center_first"

    def __call__(self, observation: str) -> str:
        available = parse_available_moves(observation)
        for move in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
            if move in available:
                return f"[{move}]"
        return "[0]"


class RandomLegalAgent:
    name = "random_legal"

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def __call__(self, observation: str) -> str:
        available = parse_available_moves(observation)
        return f"[{self._rng.choice(available)}]" if available else "[0]"


class OptimalTicTacToeAgent:
    name = "optimal_minimax"

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def __call__(self, observation: str) -> str:
        board = parse_tictactoe_board(observation)
        available = [idx for idx, value in enumerate(board) if value == ""]
        if not available:
            fallback = parse_available_moves(observation)
            return f"[{fallback[0] if fallback else 0}]"
        move = choose_minimax_move(board, self.symbol, self.opponent_symbol)
        return f"[{move}]"


def parse_available_moves(observation: str) -> list[int]:
    """Extract bracketed legal moves from a TextArena observation."""

    marker = observation.rfind("Available Moves:")
    search_space = observation[marker:] if marker >= 0 else observation
    return sorted(set(int(match) for match in re.findall(r"\[(\d+)\]", search_space)))


def parse_tictactoe_board(observation: str) -> list[str]:
    """Parse the last visible TicTacToe board from a default TextArena observation."""

    board_lines = [line for line in observation.splitlines() if "|" in line and "---" not in line]
    board_lines = board_lines[-3:]
    cells: list[str] = []
    for line in board_lines:
        for token in line.split("|"):
            token = token.strip()
            cells.append(token if token in {"X", "O"} else "")
    if len(cells) != 9:
        return [""] * 9
    return cells


def choose_minimax_move(board: list[str], symbol: str, opponent: str) -> int:
    """Return an optimal move for TicTacToe from the current player's perspective."""

    available = [idx for idx, value in enumerate(board) if value == ""]
    best_score = -2
    best_move = available[0]
    for move in _ordered_moves(available):
        board[move] = symbol
        score = _minimax(board, turn=opponent, maximizer=symbol, opponent=opponent)
        board[move] = ""
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _minimax(board: list[str], turn: str, maximizer: str, opponent: str) -> int:
    winner = _winner(board)
    if winner == maximizer:
        return 1
    if winner == opponent:
        return -1
    available = [idx for idx, value in enumerate(board) if value == ""]
    if not available:
        return 0
    scores = []
    next_turn = opponent if turn == maximizer else maximizer
    for move in available:
        board[move] = turn
        scores.append(_minimax(board, next_turn, maximizer, opponent))
        board[move] = ""
    return max(scores) if turn == maximizer else min(scores)


def _winner(board: list[str]) -> str | None:
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def _ordered_moves(moves: list[int]) -> list[int]:
    preference = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    return [move for move in preference if move in moves]


def make_agent(name: str, player_id: int, seed: int) -> TextArenaAgent:
    symbol = "O" if player_id == 0 else "X"
    if name == "optimal_minimax":
        return OptimalTicTacToeAgent(symbol=symbol)
    if name == "center_first":
        return CenterFirstAgent()
    if name == "first_available":
        return FirstAvailableAgent()
    if name == "random_legal":
        return RandomLegalAgent(seed=seed + 1009 * player_id)
    raise ValueError(f"Unknown TextArena benchmark agent: {name}")


def run_game(env_id: str, player0_name: str, player1_name: str, seed: int) -> GameResult:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    env = ta.make(env_id)
    agents = {
        0: make_agent(player0_name, player_id=0, seed=seed),
        1: make_agent(player1_name, player_id=1, seed=seed),
    }
    start = time.perf_counter()
    env.reset(num_players=2, seed=seed)
    done = False
    actions: list[dict[str, Any]] = []
    turns = 0
    while not done:
        player_id, observation = env.get_observation()
        if not isinstance(observation, str):
            observation = "\n".join(str(item) for item in observation)
        action = agents[player_id](observation)
        actions.append(
            {
                "turn": turns,
                "player_id": player_id,
                "agent": agents[player_id].name,
                "action": action,
                "available_moves": parse_available_moves(observation),
            }
        )
        done, _ = env.step(action=action)
        turns += 1
        if turns > 32:
            raise RuntimeError(f"{env_id} exceeded expected TicTacToe turn budget")

    close_result = env.close()
    rewards, game_info = close_result
    rewards = {int(pid): float(value) for pid, value in rewards.items()}
    winner = next((pid for pid, reward in rewards.items() if reward > 0), None)
    invalid_moves = {
        int(pid): bool(info.get("invalid_move", False))
        for pid, info in game_info.items()
        if isinstance(pid, int)
    }
    reason = " | ".join(
        str(info.get("reason", ""))
        for pid, info in sorted(game_info.items())
        if isinstance(pid, int)
    ).strip()
    return GameResult(
        env_id=env_id,
        seed=seed,
        player0_agent=player0_name,
        player1_agent=player1_name,
        rewards=rewards,
        winner=winner,
        draw=all(reward == 0 for reward in rewards.values()),
        turns=turns,
        invalid_moves=invalid_moves,
        reason=reason,
        actions=actions,
        runtime_seconds=time.perf_counter() - start,
    )


def run_benchmark(
    env_id: str,
    output_dir: Path,
    episodes_per_matchup: int,
    agents: list[str],
    seed: int,
) -> Path:
    try:
        import textarena as ta
        import textarena.api as ta_api
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "env_id": env_id,
            "episodes_per_matchup": episodes_per_matchup,
            "agents": agents,
            "seed": seed,
            "benchmark": "textarena_tictactoe_full_round_robin",
        },
    )
    write_json(
        output_dir / "environment_info.json",
        {
            **environment_info(),
            "textarena_version": getattr(ta, "__version__", "unknown"),
            "registered_env_count": len(getattr(ta_api, "ENV_REGISTRY", {})),
        },
    )
    games_path = output_dir / "games.jsonl"
    if games_path.exists():
        games_path.unlink()

    results: list[GameResult] = []
    for player0_agent in agents:
        for player1_agent in agents:
            for episode in range(episodes_per_matchup):
                result = run_game(
                    env_id=env_id,
                    player0_name=player0_agent,
                    player1_name=player1_agent,
                    seed=seed + episode,
                )
                results.append(result)
                append_jsonl(games_path, result)

    scoreboard = build_scoreboard(results, agents)
    matchup_rows = build_matchup_rows(results)
    write_csv(output_dir / "scoreboard.csv", scoreboard)
    write_csv(output_dir / "matchups.csv", matchup_rows)
    write_summary(output_dir / "summary.md", env_id, results, scoreboard, matchup_rows)
    return output_dir


def build_scoreboard(results: list[GameResult], agents: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in agents:
        games = wins = draws = losses = invalids = 0
        reward_sum = 0.0
        turns_sum = 0
        for result in results:
            participants = {0: result.player0_agent, 1: result.player1_agent}
            for player_id, name in participants.items():
                if name != agent:
                    continue
                games += 1
                reward = result.rewards[player_id]
                reward_sum += reward
                turns_sum += result.turns
                invalids += int(result.invalid_moves.get(player_id, False))
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
        rows.append(
            {
                "agent": agent,
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / games if games else 0.0,
                "non_loss_rate": (wins + draws) / games if games else 0.0,
                "average_reward": reward_sum / games if games else 0.0,
                "invalid_moves": invalids,
                "average_turns": turns_sum / games if games else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-row["average_reward"], -row["win_rate"], row["agent"]))


def build_matchup_rows(results: list[GameResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[GameResult]] = {}
    for result in results:
        grouped.setdefault((result.player0_agent, result.player1_agent), []).append(result)
    rows: list[dict[str, Any]] = []
    for (player0, player1), group in sorted(grouped.items()):
        rows.append(
            {
                "player0_agent": player0,
                "player1_agent": player1,
                "games": len(group),
                "player0_wins": sum(1 for result in group if result.rewards[0] > 0),
                "player1_wins": sum(1 for result in group if result.rewards[1] > 0),
                "draws": sum(1 for result in group if result.draw),
                "player0_avg_reward": sum(result.rewards[0] for result in group) / len(group),
                "player1_avg_reward": sum(result.rewards[1] for result in group) / len(group),
                "average_turns": sum(result.turns for result in group) / len(group),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    env_id: str,
    results: list[GameResult],
    scoreboard: list[dict[str, Any]],
    matchup_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# TextArena Benchmark Summary",
        "",
        f"- Environment: `{env_id}`",
        f"- Games: {len(results)}",
        f"- Invalid-move games: {sum(any(r.invalid_moves.values()) for r in results)}",
        "",
        "## Scoreboard",
        "",
        "| agent | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scoreboard:
        lines.append(
            "| {agent} | {games} | {wins} | {draws} | {losses} | {win_rate:.3f} | "
            "{non_loss_rate:.3f} | {average_reward:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Matchups",
            "",
            "| player0 | player1 | games | p0_wins | p1_wins | draws | p0_avg_reward |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in matchup_rows:
        lines.append(
            "| {player0_agent} | {player1_agent} | {games} | {player0_wins} | "
            "{player1_wins} | {draws} | {player0_avg_reward:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline TextArena benchmark.")
    parser.add_argument("--env-id", default="TicTacToe-v0")
    parser.add_argument("--episodes-per-matchup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--agents",
        default="optimal_minimax,center_first,first_available,random_legal",
        help="Comma-separated agent names.",
    )
    parser.add_argument("--output-dir", default="runs/textarena_tictactoe_full")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    output_dir = run_benchmark(
        env_id=args.env_id,
        output_dir=Path(args.output_dir),
        episodes_per_matchup=args.episodes_per_matchup,
        agents=agents,
        seed=args.seed,
    )
    print(f"TextArena benchmark artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
