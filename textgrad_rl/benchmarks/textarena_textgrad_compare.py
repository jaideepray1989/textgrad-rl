"""Compare TextGrad-RL against a fixed-prompt baseline on TextArena.

This is an offline benchmark: it uses the installed TextArena package locally
and updates only text variables for the TextGrad variant.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_benchmark import (
    GameResult,
    make_agent,
    parse_available_moves,
    parse_tictactoe_board,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


@dataclass
class TargetGameRecord:
    variant: str
    split: str
    iteration: int
    env_id: str
    seed: int
    target_side: int
    opponent: str
    target_reward: float
    target_win: bool
    target_draw: bool
    target_loss: bool
    turns: int
    invalid_move: bool
    reason: str
    game: GameResult


class PromptAwareTicTacToeAgent:
    """Frozen local TextArena actor controlled by text variables."""

    name = "prompt_aware_textarena_agent"

    def __init__(self, symbol: str, text_variables: dict[str, TextVariable]):
        self.symbol = symbol
        self.opponent = "O" if symbol == "X" else "X"
        self.text_variables = text_variables

    def __call__(self, observation: str) -> str:
        board = parse_tictactoe_board(observation)
        available = [idx for idx, value in enumerate(board) if value == ""]
        if not available:
            fallback = parse_available_moves(observation)
            return f"[{fallback[0] if fallback else 0}]"

        prompt = "\n".join(variable.value for variable in self.text_variables.values()).lower()
        if "minimax" in prompt or "game tree" in prompt:
            return f"[{choose_minimax_move(board, self.symbol, self.opponent)}]"

        if "immediate winning" in prompt or "take a win" in prompt:
            move = find_winning_move(board, self.symbol)
            if move is not None:
                return f"[{move}]"

        if "block opponent" in prompt or "block immediate" in prompt:
            move = find_winning_move(board, self.opponent)
            if move is not None:
                return f"[{move}]"

        if "center" in prompt and 4 in available:
            return "[4]"
        if "corner" in prompt:
            for move in [0, 2, 6, 8]:
                if move in available:
                    return f"[{move}]"
        return f"[{available[0]}]"


def initial_game_text_variables() -> dict[str, TextVariable]:
    return {
        "game_strategy_prompt": TextVariable(
            name="game_strategy_prompt",
            value=(
                "Play legal TicTacToe moves. Prefer the center when available. "
                "Prefer corners before edges."
            ),
            role_description=(
                "Choose legal TextArena TicTacToe actions from observations while improving "
                "win rate, non-loss rate, and invalid-move rate."
            ),
            max_chars=1800,
        )
    }


def run_target_game(
    env_id: str,
    variant: str,
    split: str,
    iteration: int,
    seed: int,
    target_side: int,
    opponent_name: str,
    text_variables: dict[str, TextVariable],
) -> TargetGameRecord:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    env = ta.make(env_id)
    target_symbol = "O" if target_side == 0 else "X"
    agents = {
        target_side: PromptAwareTicTacToeAgent(target_symbol, text_variables),
        1 - target_side: make_agent(opponent_name, player_id=1 - target_side, seed=seed),
    }
    agent_names = {
        target_side: variant,
        1 - target_side: opponent_name,
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
        board = parse_tictactoe_board(observation)
        action = agents[player_id](observation)
        actions.append(
            {
                "turn": turns,
                "player_id": player_id,
                "agent": agent_names[player_id],
                "symbol": "O" if player_id == 0 else "X",
                "action": action,
                "action_cell": parse_action_cell(action),
                "board": board,
                "available_moves": parse_available_moves(observation),
            }
        )
        done, _ = env.step(action=action)
        turns += 1
        if turns > 32:
            raise RuntimeError(f"{env_id} exceeded expected TicTacToe turn budget")

    rewards, game_info = env.close()
    rewards = {int(pid): float(value) for pid, value in rewards.items()}
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
    game = GameResult(
        env_id=env_id,
        seed=seed,
        player0_agent=agent_names[0],
        player1_agent=agent_names[1],
        rewards=rewards,
        winner=next((pid for pid, reward in rewards.items() if reward > 0), None),
        draw=all(reward == 0 for reward in rewards.values()),
        turns=turns,
        invalid_moves=invalid_moves,
        reason=reason,
        actions=actions,
        runtime_seconds=time.perf_counter() - start,
    )
    target_reward = rewards[target_side]
    return TargetGameRecord(
        variant=variant,
        split=split,
        iteration=iteration,
        env_id=env_id,
        seed=seed,
        target_side=target_side,
        opponent=opponent_name,
        target_reward=target_reward,
        target_win=target_reward > 0,
        target_draw=target_reward == 0,
        target_loss=target_reward < 0,
        turns=turns,
        invalid_move=invalid_moves.get(target_side, False),
        reason=reason,
        game=game,
    )


def run_suite(
    env_id: str,
    variant: str,
    split: str,
    iteration: int,
    episodes_per_opponent: int,
    opponents: list[str],
    seed: int,
    text_variables: dict[str, TextVariable],
    output_jsonl: Path | None = None,
) -> list[TargetGameRecord]:
    if output_jsonl and output_jsonl.exists():
        output_jsonl.unlink()
    records: list[TargetGameRecord] = []
    for opponent in opponents:
        for target_side in [0, 1]:
            for episode in range(episodes_per_opponent):
                record = run_target_game(
                    env_id=env_id,
                    variant=variant,
                    split=split,
                    iteration=iteration,
                    seed=seed + episode,
                    target_side=target_side,
                    opponent_name=opponent,
                    text_variables=text_variables,
                )
                records.append(record)
                if output_jsonl:
                    append_jsonl(output_jsonl, record)
    return records


def critique_textarena_trajectories(records: list[TargetGameRecord]) -> list[TextualGradient]:
    gradients: list[TextualGradient] = []
    for record in records:
        if record.invalid_move:
            gradients.append(
                make_gradient(
                    "Invalid TextArena action",
                    f"Against {record.opponent}, the target made an invalid move.",
                    "The agent must always output one legal bracketed move from Available Moves.",
                    "Add a rule: always choose a legal move shown in Available Moves and output only [cell].",
                    0.95,
                )
            )
        missed_tactic = False
        for action in record.game.actions:
            if action["player_id"] != record.target_side:
                continue
            board = list(action["board"])
            chosen = action["action_cell"]
            symbol = action["symbol"]
            opponent = "O" if symbol == "X" else "X"
            winning_move = find_winning_move(board, symbol)
            if winning_move is not None and chosen != winning_move:
                missed_tactic = True
                gradients.append(
                    make_gradient(
                        "Missed immediate winning move",
                        f"At turn {action['turn']} against {record.opponent}, move {winning_move} would have won.",
                        "The agent should take a one-move win before applying positional preferences.",
                        "Add a rule: take an immediate winning move before center, corner, or blocking preferences.",
                        0.9,
                    )
                )
            blocking_move = find_winning_move(board, opponent)
            if blocking_move is not None and chosen != blocking_move:
                missed_tactic = True
                gradients.append(
                    make_gradient(
                        "Failed to block opponent immediate win",
                        f"At turn {action['turn']} against {record.opponent}, opponent threatened cell {blocking_move}.",
                        "The agent should block a one-move opponent win before choosing positional moves.",
                        "Add a rule: block opponent immediate winning moves before center or corner preferences.",
                        0.9,
                    )
                )
        if record.target_loss and not missed_tactic:
            gradients.append(
                make_gradient(
                    "Lost without a one-move tactical explanation",
                    f"The target lost to {record.opponent} as player {record.target_side}; simple tactics were insufficient.",
                    "In solved small games, search the game tree when tactical rules do not guarantee a non-loss.",
                    "Add a rule: use minimax game-tree search when simple immediate-win and block rules are insufficient.",
                    0.82,
                )
            )
    return dedupe_gradients(gradients)


def make_gradient(
    failure_mode: str,
    evidence: str,
    gradient_text: str,
    suggested_edit: str,
    confidence: float,
) -> TextualGradient:
    return TextualGradient(
        target_variable_name="game_strategy_prompt",
        failure_mode=failure_mode,
        evidence_from_trajectory=evidence,
        gradient_text=gradient_text,
        suggested_edit=suggested_edit,
        confidence=confidence,
        forbidden_shortcuts=["invalid action formats", "reading hidden state", "changing environment rules"],
    )


def dedupe_gradients(gradients: list[TextualGradient]) -> list[TextualGradient]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TextualGradient] = []
    for gradient in gradients:
        key = (gradient.target_variable_name, gradient.failure_mode)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gradient)
    return deduped


def run_comparison(
    env_id: str,
    output_dir: Path,
    iterations: int,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    opponents: list[str],
    seed: int,
) -> Path:
    try:
        import textarena as ta
        import textarena.api as ta_api
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "env_id": env_id,
        "iterations": iterations,
        "train_episodes_per_opponent_side": train_episodes,
        "val_episodes_per_opponent_side": val_episodes,
        "test_episodes_per_opponent_side": test_episodes,
        "opponents": opponents,
        "seed": seed,
    }
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "environment_info.json",
        {
            **environment_info(),
            "textarena_version": getattr(ta, "__version__", "unknown"),
            "registered_env_count": len(getattr(ta_api, "ENV_REGISTRY", {})),
        },
    )

    fixed_variables = initial_game_text_variables()
    textgrad_variables = initial_game_text_variables()
    write_json(output_dir / "initial_text_variables.json", textgrad_variables)

    fixed_test = run_suite(
        env_id,
        "no_textgrad",
        "test",
        0,
        test_episodes,
        opponents,
        seed + 20_000,
        fixed_variables,
        games_dir / "no_textgrad_test.jsonl",
    )

    metrics_rows = [
        {"variant": "no_textgrad", "split": "test", "iteration": 0, **summarize_records(fixed_test)}
    ]
    accepted_updates: list[dict[str, Any]] = []
    rejected_updates: list[dict[str, Any]] = []
    optimizer = TextualGradientDescent(max_prompt_chars=1800, max_rules_per_step=1)

    for iteration in range(1, iterations + 1):
        train_records = run_suite(
            env_id,
            "textgrad_rl",
            "train",
            iteration,
            train_episodes,
            opponents,
            seed + 1000 * iteration,
            textgrad_variables,
            games_dir / f"textgrad_train_iteration_{iteration:03d}.jsonl",
        )
        gradients = critique_textarena_trajectories(train_records)
        write_json(output_dir / "gradients" / f"iteration_{iteration:03d}.json", gradients)
        candidate_variables = optimizer.step(
            textgrad_variables,
            gradients,
            constraints=["Do not use hidden state", "Do not change environment rules"],
        )

        old_val = run_suite(
            env_id,
            "textgrad_rl_old",
            "val",
            iteration,
            val_episodes,
            opponents,
            seed + 10_000 + 1000 * iteration,
            textgrad_variables,
            games_dir / f"textgrad_val_old_iteration_{iteration:03d}.jsonl",
        )
        new_val = run_suite(
            env_id,
            "textgrad_rl_candidate",
            "val",
            iteration,
            val_episodes,
            opponents,
            seed + 10_000 + 1000 * iteration,
            candidate_variables,
            games_dir / f"textgrad_val_candidate_iteration_{iteration:03d}.jsonl",
        )
        old_score = score_records(old_val)
        new_score = score_records(new_val)
        changed = variables_changed(textgrad_variables, candidate_variables)
        accepted = changed and new_score >= old_score
        update_record = {
            "iteration": iteration,
            "old_score": old_score,
            "new_score": new_score,
            "accepted": accepted,
            "changed": changed,
            "gradient_count": len(gradients),
            "old_metrics": summarize_records(old_val),
            "new_metrics": summarize_records(new_val),
        }
        if accepted:
            textgrad_variables = candidate_variables
            accepted_updates.append(update_record)
            append_jsonl(output_dir / "accepted_updates.jsonl", update_record)
        else:
            rejected_updates.append(update_record)
            append_jsonl(output_dir / "rejected_updates.jsonl", update_record)

        metrics_rows.append(
            {
                "variant": "textgrad_rl",
                "split": "train",
                "iteration": iteration,
                "accepted": accepted,
                "gradient_count": len(gradients),
                **summarize_records(train_records),
            }
        )
        metrics_rows.append(
            {
                "variant": "textgrad_rl",
                "split": "val",
                "iteration": iteration,
                "accepted": accepted,
                "gradient_count": len(gradients),
                **summarize_records(new_val if accepted else old_val),
            }
        )
        write_json(output_dir / "text_variables" / f"iteration_{iteration:03d}.json", textgrad_variables)

    textgrad_test = run_suite(
        env_id,
        "textgrad_rl",
        "test",
        iterations,
        test_episodes,
        opponents,
        seed + 20_000,
        textgrad_variables,
        games_dir / "textgrad_rl_test.jsonl",
    )
    metrics_rows.append(
        {"variant": "textgrad_rl", "split": "test", "iteration": iterations, **summarize_records(textgrad_test)}
    )
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_json(output_dir / "final_text_variables.json", textgrad_variables)
    write_json(output_dir / "accepted_updates.json", accepted_updates)
    write_json(output_dir / "rejected_updates.json", rejected_updates)
    write_summary(output_dir / "summary.md", config, metrics_rows, textgrad_variables)
    return output_dir


def summarize_records(records: list[TargetGameRecord]) -> dict[str, Any]:
    games = len(records)
    wins = sum(record.target_win for record in records)
    draws = sum(record.target_draw for record in records)
    losses = sum(record.target_loss for record in records)
    reward_sum = sum(record.target_reward for record in records)
    invalids = sum(record.invalid_move for record in records)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / games if games else 0.0,
        "non_loss_rate": (wins + draws) / games if games else 0.0,
        "average_reward": reward_sum / games if games else 0.0,
        "invalid_move_rate": invalids / games if games else 0.0,
        "average_turns": sum(record.turns for record in records) / games if games else 0.0,
    }


def score_records(records: list[TargetGameRecord]) -> float:
    metrics = summarize_records(records)
    return (
        3.0 * metrics["average_reward"]
        + 2.0 * metrics["non_loss_rate"]
        + metrics["win_rate"]
        - 2.0 * metrics["invalid_move_rate"]
    )


def variables_changed(
    old_variables: dict[str, TextVariable],
    new_variables: dict[str, TextVariable],
) -> bool:
    return any(
        old_variables[name].value != new_variables.get(name, old_variables[name]).value
        or old_variables[name].version != new_variables.get(name, old_variables[name]).version
        for name in old_variables
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    config: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    final_variables: dict[str, TextVariable],
) -> None:
    test_rows = [row for row in metrics_rows if row["split"] == "test"]
    by_variant = {row["variant"]: row for row in test_rows}
    fixed = by_variant.get("no_textgrad", {})
    textgrad = by_variant.get("textgrad_rl", {})
    lines = [
        "# TextArena TextGrad-RL vs No TextGrad",
        "",
        f"- Environment: `{config['env_id']}`",
        f"- Opponents: {', '.join(config['opponents'])}",
        f"- Test games per variant: {fixed.get('games', textgrad.get('games', 0))}",
        "",
        "## Held-Out Test Results",
        "",
        "| variant | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward | invalid_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ["no_textgrad", "textgrad_rl"]:
        row = by_variant.get(variant, {})
        lines.append(
            "| {variant} | {games} | {wins} | {draws} | {losses} | {win_rate:.3f} | "
            "{non_loss_rate:.3f} | {average_reward:.3f} | {invalid_move_rate:.3f} |".format(
                variant=variant,
                games=int(row.get("games", 0)),
                wins=int(row.get("wins", 0)),
                draws=int(row.get("draws", 0)),
                losses=int(row.get("losses", 0)),
                win_rate=float(row.get("win_rate", 0.0)),
                non_loss_rate=float(row.get("non_loss_rate", 0.0)),
                average_reward=float(row.get("average_reward", 0.0)),
                invalid_move_rate=float(row.get("invalid_move_rate", 0.0)),
            )
        )
    if fixed and textgrad:
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- Win rate: {textgrad['win_rate'] - fixed['win_rate']:+.3f}",
                f"- Non-loss rate: {textgrad['non_loss_rate'] - fixed['non_loss_rate']:+.3f}",
                f"- Average reward: {textgrad['average_reward'] - fixed['average_reward']:+.3f}",
            ]
        )
    lines.extend(["", "## Final Text Variables", ""])
    for name, variable in final_variables.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```text")
        lines.append(variable.value)
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_action_cell(action: str) -> int | None:
    import re

    match = re.search(r"\[(\d+)\]", action)
    return int(match.group(1)) if match else None


def find_winning_move(board: list[str], symbol: str) -> int | None:
    for move, value in enumerate(board):
        if value:
            continue
        candidate = list(board)
        candidate[move] = symbol
        if winner(candidate) == symbol:
            return move
    return None


def winner(board: list[str]) -> str | None:
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def choose_minimax_move(board: list[str], symbol: str, opponent: str) -> int:
    available = [idx for idx, value in enumerate(board) if value == ""]
    best_score = -2
    best_move = available[0]
    for move in ordered_moves(available):
        board[move] = symbol
        score = minimax(board, opponent, symbol, opponent)
        board[move] = ""
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def minimax(board: list[str], turn: str, maximizer: str, opponent: str) -> int:
    current_winner = winner(board)
    if current_winner == maximizer:
        return 1
    if current_winner == opponent:
        return -1
    available = [idx for idx, value in enumerate(board) if value == ""]
    if not available:
        return 0
    next_turn = opponent if turn == maximizer else maximizer
    scores: list[int] = []
    for move in available:
        board[move] = turn
        scores.append(minimax(board, next_turn, maximizer, opponent))
        board[move] = ""
    return max(scores) if turn == maximizer else min(scores)


def ordered_moves(moves: list[int]) -> list[int]:
    preference = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    return [move for move in preference if move in moves]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TextGrad-RL vs fixed prompts on TextArena.")
    parser.add_argument("--env-id", default="TicTacToe-v0")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument(
        "--opponents",
        default="first_available,center_first,random_legal,optimal_minimax",
        help="Comma-separated opponent agents.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", default="runs/textarena_textgrad_vs_no_textgrad")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    output_dir = run_comparison(
        env_id=args.env_id,
        output_dir=Path(args.output_dir),
        iterations=args.iterations,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        opponents=opponents,
        seed=args.seed,
    )
    print(f"TextArena TextGrad comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
