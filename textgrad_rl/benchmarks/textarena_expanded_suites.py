"""Expanded TextArena benchmark suites for policy iteration and real SLMs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_multienv_compare import (
    DEFAULT_ENVS,
    EpisodeRecord,
    num_players_for_env,
    run_records,
    summarize,
)
from textgrad_rl.benchmarks.textarena_paper_suite import initial_modular_variables, slug_env
from textgrad_rl.benchmarks.textarena_policy_iteration import (
    POLICY_ABLATION_GATE_MODES,
    TextPPOConfig,
    run_policy_ablation_once,
    clipped_text_surrogate,
    run_policy_iteration_once,
    run_textgrad_ppo_once,
    text_behavior_ratio,
)
from textgrad_rl.benchmarks.textarena_slm_compare import OpenAICompatibleChatModel
from textgrad_rl.benchmarks.textarena_textgrad_plus import run_baseline_once, run_textgrad_plus_once
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


DIFFICULTY_ENVS = [
    "GuessTheNumber-v0-hardcore",
    "FrozenLake-v0-random",
    "FrozenLake-v0-hardcore",
    "Nim-v0-medium",
    "Nim-v0-large",
    "TowerOfHanoi-v0-medium",
    "TowerOfHanoi-v0-hard",
    "Blackjack-v0-long",
    "Bandit-v0-hard",
    "Mastermind-v0-hard",
]

PUZZLE_SLM_ENVS = [
    "Wordle-v0",
    "Hangman-v0",
    "Minesweeper-v0-small",
    "Sudoku-v0-very-easy",
]

SOCIAL_SLM_ENVS = [
    "KuhnPoker-v0-short",
    "IteratedPrisonersDilemma-v0",
    "SimpleNegotiation-v0-short",
]

REAL_SLM_ENVS = [
    "GuessTheNumber-v0",
    "FrozenLake-v0",
    "Blackjack-v0",
    "Nim-v0",
    "Mastermind-v0",
]

DIFFICULTY_METHODS = [
    "fixed_prompt",
    "modular_textgrad",
    "textgrad_rl_plus",
    "textgrad_policy_iteration",
    "textgrad_ppo",
    "textgrad_rl_no_gate",
    "textgrad_rl_train_val",
    "textgrad_rl_kl_gate",
    "textgrad_rl_clipped_surrogate",
    "textgrad_rl_ppo",
]

SLM_METHODS = [
    "fixed_prompt_slm",
    "scalar_prompt_search_slm",
    "modular_textgrad_slm",
    "textgrad_rl_plus_slm",
    "textgrad_policy_iteration_slm",
    "textgrad_ppo_slm",
    "textgrad_rl_no_gate_slm",
    "textgrad_rl_train_val_slm",
    "textgrad_rl_kl_gate_slm",
    "textgrad_rl_clipped_surrogate_slm",
    "textgrad_rl_ppo_slm",
]

SLM_ABLATION_GATE_MODES = {
    "textgrad_rl_no_gate_slm": "no_gate",
    "textgrad_rl_train_val_slm": "train_val",
    "textgrad_rl_kl_gate_slm": "kl_gate",
    "textgrad_rl_clipped_surrogate_slm": "clipped_surrogate",
    "textgrad_rl_ppo_slm": "ppo",
    "textgrad_policy_iteration_slm": "train_val",
    "textgrad_ppo_slm": "ppo",
}

SLM_NUM_PLAYERS = {
    "Wordle-v0": 1,
    "Hangman-v0": 1,
    "Minesweeper-v0-small": 1,
    "Sudoku-v0-very-easy": 1,
    "KuhnPoker-v0-short": 2,
    "IteratedPrisonersDilemma-v0": 2,
    "SimpleNegotiation-v0-short": 2,
    "GuessTheNumber-v0": 1,
    "FrozenLake-v0": 1,
    "Blackjack-v0": 1,
    "Nim-v0": 2,
    "Mastermind-v0": 1,
}

SLM_MAX_TURNS = {
    "Wordle-v0": 8,
    "Hangman-v0": 12,
    "Minesweeper-v0-small": 12,
    "Sudoku-v0-very-easy": 12,
    "KuhnPoker-v0-short": 60,
    "IteratedPrisonersDilemma-v0": 50,
    "SimpleNegotiation-v0-short": 50,
    "GuessTheNumber-v0": 12,
    "FrozenLake-v0": 20,
    "Blackjack-v0": 15,
    "Nim-v0": 30,
    "Mastermind-v0": 12,
}

SLM_HINTS = {
    "Wordle-v0": "Use feedback to eliminate letters. Return one lowercase 5-letter guess in brackets.",
    "Hangman-v0": "Guess common letters first. Return one bracketed letter unless you know the full word.",
    "Minesweeper-v0-small": "Reveal safe-looking cells by row and column in brackets. Avoid repeating revealed cells.",
    "Sudoku-v0-very-easy": "Return one legal placement in the requested coordinate/value format from the grid.",
    "KuhnPoker-v0-short": "Use conservative Kuhn poker strategy: bet strong cards, check weak cards, call selectively.",
    "IteratedPrisonersDilemma-v0": "Prefer cooperation unless the opponent defects repeatedly; use the exact decision token when asked.",
    "SimpleNegotiation-v0-short": "Make mutually beneficial offers using the required offer token format.",
    "GuessTheNumber-v0": "Maintain lower and upper bounds from higher/lower hints and guess the midpoint.",
    "FrozenLake-v0": "Plan a safe path to G while avoiding H; output one bracketed direction.",
    "Blackjack-v0": "Stand on 17 or higher; hit below 17 unless bust risk is obvious.",
    "Nim-v0": "Use the xor strategy: move to a zero nim-sum when possible.",
    "Mastermind-v0": "Maintain candidate codes and eliminate guesses inconsistent with black/white feedback.",
}


@dataclass
class SLMTextArenaRecord:
    suite: str
    env_id: str
    variant: str
    split: str
    seed: int
    model: str
    target_player: int
    reward: float
    success: bool
    invalid_move: bool
    truncated: bool
    turns: int
    reason: str
    actions: list[str]
    raw_outputs: list[str]
    runtime_seconds: float


class GenericTextArenaSLMAgent:
    def __init__(self, env_id: str, model: OpenAICompatibleChatModel, variables: dict[str, TextVariable]) -> None:
        self.env_id = env_id
        self.model = model
        self.variables = variables

    def act(self, observation: str, player_id: int) -> tuple[str, str]:
        raw = self.model.complete(self._prompt(observation, player_id))
        return normalize_textarena_action(raw, self.env_id), raw

    def _prompt(self, observation: str, player_id: int) -> str:
        variable_text = "\n\n".join(
            f"{variable.name} ({variable.role_description}):\n{variable.clipped_value()}"
            for variable in self.variables.values()
        )
        hint = SLM_HINTS.get(self.env_id, "Read the rules and output one legal action.")
        return (
            f"TEXT VARIABLES:\n{variable_text}\n\n"
            f"ENVIRONMENT: {self.env_id}\nPLAYER: {player_id}\nHINT: {hint}\n\n"
            f"OBSERVATION:\n{observation}\n\n"
            "Return exactly one legal TextArena action. Prefer a bracketed action if the game defines one."
        )


def normalize_textarena_action(raw_output: str, env_id: str) -> str:
    bracketed = re.findall(r"\[[^\]]+\]", raw_output)
    if bracketed:
        action = bracketed[0]
    else:
        cleaned = raw_output.strip().splitlines()[0] if raw_output.strip() else "[0]"
        action = cleaned[:200]
    if "GuessTheNumber" in env_id:
        numbers = [int(value) for value in re.findall(r"-?\d+", action)]
        if numbers:
            return f"[{max(1, min(100, numbers[0]))}]"
        return "[10]"
    return action


def initial_slm_variables(env_ids: list[str]) -> dict[str, TextVariable]:
    variables = {
        "general_textarena_slm_policy": TextVariable(
            name="general_textarena_slm_policy",
            value=(
                "Follow the visible TextArena rules. Track state across observations, avoid repeating "
                "invalid actions, and output only one legal action."
            ),
            role_description="General frozen-SLM TextArena policy.",
            max_chars=1800,
        )
    }
    for env_id in env_ids:
        name = f"{slug_env(env_id)}_slm_policy"
        variables[name] = TextVariable(
            name=name,
            value=f"For {env_id}: {SLM_HINTS.get(env_id, 'read the game rules and choose a legal action.')}",
            role_description=f"Environment-specific frozen-SLM policy for {env_id}.",
            max_chars=1800,
        )
    return variables


def slm_gradients_from_records(records: list[SLMTextArenaRecord]) -> list[TextualGradient]:
    gradients: list[TextualGradient] = []
    for env_id in sorted({record.env_id for record in records}):
        group = [record for record in records if record.env_id == env_id]
        failures = [record for record in group if not record.success or record.invalid_move or record.truncated]
        if not failures:
            continue
        repeated = sum(has_repeated_actions(record.actions) for record in group)
        target = f"{slug_env(env_id)}_slm_policy"
        gradients.append(
            TextualGradient(
                target_variable_name=target,
                failure_mode=f"{env_id} frozen-SLM trajectories had low return",
                evidence_from_trajectory=(
                    f"{len(failures)}/{len(group)} episodes failed, were invalid, or hit the turn cap; "
                    f"{repeated} episodes repeated actions."
                ),
                gradient_text="Improve legal action selection and state tracking from trajectory feedback.",
                suggested_edit=(
                    "Add a rule: Track the latest observation, obey the exact action format, avoid repeated "
                    f"invalid/no-progress actions, and use this environment hint: {SLM_HINTS.get(env_id, 'choose a legal action')}."
                ),
                confidence=0.75,
                forbidden_shortcuts=["hidden state", "environment mutation", "invalid action formats"],
            )
        )
    return gradients


def has_repeated_actions(actions: list[str]) -> bool:
    return len(actions) != len(set(actions))


def slm_record_score(record: SLMTextArenaRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.5 * (1.0 if record.invalid_move else 0.0)
        - 0.25 * (1.0 if record.truncated else 0.0)
        - 0.005 * record.turns
    )


def run_slm_episode(
    suite: str,
    env_id: str,
    variant: str,
    split: str,
    seed: int,
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
) -> SLMTextArenaRecord:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    num_players = SLM_NUM_PLAYERS[env_id]
    target_player = 0
    env = ta.make(env_id)
    env.reset(num_players=num_players, seed=seed)
    agent = GenericTextArenaSLMAgent(env_id, model, variables)
    actions: list[str] = []
    raw_outputs: list[str] = []
    done = False
    truncated = False
    turns = 0
    start = time.perf_counter()
    max_turns = SLM_MAX_TURNS[env_id]
    while not done and turns < max_turns:
        pid, observation = env.get_observation()
        if not isinstance(observation, str):
            observation = "\n".join(str(item) for item in observation)
        try:
            action, raw = agent.act(observation, pid)
        except Exception as exc:
            action = "[0]"
            raw = f"REQUEST_FAILED: {exc}"
        actions.append(f"p{pid}:{action}")
        raw_outputs.append(raw)
        done, _ = env.step(action)
        turns += 1
    if not done:
        truncated = True
    rewards, game_info = env.close()
    info = game_info.get(target_player, {}) if isinstance(game_info, dict) else {}
    reward = float(rewards.get(target_player, 0.0)) if isinstance(rewards, dict) else 0.0
    return SLMTextArenaRecord(
        suite=suite,
        env_id=env_id,
        variant=variant,
        split=split,
        seed=seed,
        model=model.model,
        target_player=target_player,
        reward=reward,
        success=reward >= 1.0,
        invalid_move=bool(info.get("invalid_move", False)),
        truncated=truncated,
        turns=turns,
        reason=str(info.get("reason", "")),
        actions=actions,
        raw_outputs=raw_outputs,
        runtime_seconds=time.perf_counter() - start,
    )


def run_slm_records(
    suite: str,
    env_ids: list[str],
    variant: str,
    split: str,
    seeds_per_env: int,
    seed: int,
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
    path: Path,
) -> list[SLMTextArenaRecord]:
    if path.exists():
        path.unlink()
    records: list[SLMTextArenaRecord] = []
    for env_id in env_ids:
        for idx in range(seeds_per_env):
            record = run_slm_episode(suite, env_id, variant, split, seed + idx, model, variables)
            records.append(record)
            append_jsonl(path, record)
    return records


def summarize_slm(records: list[SLMTextArenaRecord]) -> dict[str, Any]:
    n = len(records)
    return {
        "episodes": n,
        "average_reward": sum(record.reward for record in records) / n if n else 0.0,
        "success_rate": sum(record.success for record in records) / n if n else 0.0,
        "invalid_move_rate": sum(record.invalid_move for record in records) / n if n else 0.0,
        "truncation_rate": sum(record.truncated for record in records) / n if n else 0.0,
        "average_turns": sum(record.turns for record in records) / n if n else 0.0,
        "average_score": sum(slm_record_score(record) for record in records) / n if n else 0.0,
    }


def slm_score(records: list[SLMTextArenaRecord]) -> float:
    return sum(slm_record_score(record) for record in records) / len(records) if records else 0.0


def slm_env_baselines(records: list[SLMTextArenaRecord]) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for env_id in sorted({record.env_id for record in records}):
        group = [slm_record_score(record) for record in records if record.env_id == env_id]
        baselines[env_id] = sum(group) / len(group) if group else 0.0
    return baselines


def slm_fallback_advantage(record: SLMTextArenaRecord) -> float:
    if record.success and not record.invalid_move and not record.truncated:
        return 0.5
    penalty = 0.5
    if record.invalid_move:
        penalty += 0.5
    if record.truncated:
        penalty += 0.25
    if has_repeated_actions(record.actions):
        penalty += 0.25
    return -penalty


def slm_ppo_metrics(
    old_records: list[SLMTextArenaRecord],
    new_records: list[SLMTextArenaRecord],
    baseline_records: list[SLMTextArenaRecord],
    config: TextPPOConfig,
) -> dict[str, Any]:
    paired = list(zip(old_records, new_records))
    baselines = slm_env_baselines(baseline_records)
    ratios: list[float] = []
    old_surrogates: list[float] = []
    clipped_surrogates: list[float] = []
    score_deltas: list[float] = []
    action_changes: list[float] = []
    for old, new in paired:
        old_score = slm_record_score(old)
        new_score = slm_record_score(new)
        advantage = old_score - baselines.get(old.env_id, slm_score(baseline_records))
        if abs(advantage) < 1e-12:
            advantage = slm_fallback_advantage(old)
        ratio = text_behavior_ratio(old_score, new_score, advantage, config.score_scale)
        ratios.append(ratio)
        old_surrogates.append(advantage)
        clipped_surrogates.append(clipped_text_surrogate(advantage, ratio, config.clip_epsilon))
        score_deltas.append(new_score - old_score)
        action_changes.append(0.0 if old.actions == new.actions else 1.0)

    old_metrics = summarize_slm(old_records)
    new_metrics = summarize_slm(new_records)
    n = len(paired)
    if not n:
        return {
            "paired_episodes": 0,
            "old_surrogate": 0.0,
            "clipped_surrogate": 0.0,
            "surrogate_delta": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "mean_score_delta": 0.0,
            "mean_ratio": 1.0,
            "action_change_rate": 0.0,
            "invalid_delta": new_metrics["invalid_move_rate"] - old_metrics["invalid_move_rate"],
            "truncation_delta": new_metrics["truncation_rate"] - old_metrics["truncation_rate"],
            "turn_delta": new_metrics["average_turns"] - old_metrics["average_turns"],
        }
    old_surrogate = sum(old_surrogates) / n
    clipped_surrogate = sum(clipped_surrogates) / n
    return {
        "paired_episodes": n,
        "old_surrogate": old_surrogate,
        "clipped_surrogate": clipped_surrogate,
        "surrogate_delta": clipped_surrogate - old_surrogate,
        "approx_kl": sum(0.5 * math.log(ratio) ** 2 for ratio in ratios) / n,
        "clip_fraction": sum(
            ratio < 1.0 - config.clip_epsilon or ratio > 1.0 + config.clip_epsilon
            for ratio in ratios
        )
        / n,
        "mean_score_delta": sum(score_deltas) / n,
        "mean_ratio": sum(ratios) / n,
        "action_change_rate": sum(action_changes) / n,
        "invalid_delta": new_metrics["invalid_move_rate"] - old_metrics["invalid_move_rate"],
        "truncation_delta": new_metrics["truncation_rate"] - old_metrics["truncation_rate"],
        "turn_delta": new_metrics["average_turns"] - old_metrics["average_turns"],
    }


def slm_ppo_objective(metrics: dict[str, Any], val_delta: float) -> float:
    return (
        float(metrics["surrogate_delta"])
        + 0.25 * val_delta
        - 0.5 * float(metrics["approx_kl"])
        - max(0.0, float(metrics["invalid_delta"]))
        - max(0.0, float(metrics["truncation_delta"]))
        - 0.005 * max(0.0, float(metrics["turn_delta"]))
    )


def scalar_slm_update(
    variables: dict[str, TextVariable],
    train_records: list[SLMTextArenaRecord],
) -> dict[str, TextVariable]:
    import copy

    updated = copy.deepcopy(variables)
    variable = updated["general_textarena_slm_policy"]
    invalid = sum(record.invalid_move for record in train_records)
    truncated = sum(record.truncated for record in train_records)
    rule = (
        "Before answering, copy the exact action format from the current game instructions; "
        "if a previous action failed or made no progress, choose a different legal action."
    )
    if rule.lower() not in variable.value.lower():
        if "Learned rules:" not in variable.value:
            variable.value = variable.value.rstrip() + "\n\nLearned rules:"
        variable.value += f"\n- {rule}"
        variable.version += 1
        variable.gradient_history.append(f"scalar_slm_update: invalid={invalid}, truncated={truncated}")
    return updated


def build_slm_candidate(
    method: str,
    variables: dict[str, TextVariable],
    train_records: list[SLMTextArenaRecord],
    gradients: list[TextualGradient],
) -> dict[str, TextVariable]:
    if method == "scalar_prompt_search_slm":
        return scalar_slm_update(variables, train_records)
    return TextualGradientDescent(max_prompt_chars=2400, max_rules_per_step=20).step(
        variables,
        gradients,
        constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
    )


def slm_update_accepted(
    method: str,
    train: list[SLMTextArenaRecord],
    train_candidate: list[SLMTextArenaRecord],
    val_old: list[SLMTextArenaRecord],
    val_candidate: list[SLMTextArenaRecord],
    variables: dict[str, TextVariable],
    candidate: dict[str, TextVariable],
) -> tuple[bool, dict[str, Any]]:
    old_score = slm_score(val_old)
    new_score = slm_score(val_candidate)
    train_old_score = slm_score(train)
    train_new_score = slm_score(train_candidate)
    old_metrics = summarize_slm(val_old)
    new_metrics = summarize_slm(val_candidate)
    changed = any(variables[name].value != candidate.get(name, variables[name]).value for name in variables)
    ppo_config: TextPPOConfig | None = None
    ppo_metrics: dict[str, Any] | None = None
    ppo_objective: float | None = None
    gate_mode = SLM_ABLATION_GATE_MODES.get(method)
    if method in {"scalar_prompt_search_slm", "modular_textgrad_slm"}:
        accepted = changed and new_score >= old_score
    elif method == "textgrad_rl_plus_slm":
        accepted = (
            changed
            and new_score > old_score
            and new_metrics["invalid_move_rate"] <= old_metrics["invalid_move_rate"]
            and new_metrics["truncation_rate"] <= old_metrics["truncation_rate"]
        )
    elif gate_mode is not None:
        ppo_config = TextPPOConfig(target_kl=0.2, score_scale=5.0)
        ppo_metrics = slm_ppo_metrics(train + val_old, train_candidate + val_candidate, train, ppo_config)
        ppo_objective = slm_ppo_objective(ppo_metrics, new_score - old_score)
        if gate_mode == "no_gate":
            accepted = changed
        elif gate_mode == "train_val":
            accepted = (
                changed
                and new_score > old_score
                and train_new_score >= train_old_score
                and new_metrics["invalid_move_rate"] <= old_metrics["invalid_move_rate"]
                and new_metrics["truncation_rate"] <= old_metrics["truncation_rate"]
            )
        elif gate_mode == "kl_gate":
            accepted = changed and new_score > old_score and ppo_metrics["approx_kl"] <= ppo_config.target_kl
        elif gate_mode == "clipped_surrogate":
            accepted = changed and ppo_metrics["surrogate_delta"] > ppo_config.min_surrogate_delta
        elif gate_mode == "ppo":
            accepted = (
                changed
                and new_score > old_score
                and train_new_score >= train_old_score
                and ppo_metrics["surrogate_delta"] > ppo_config.min_surrogate_delta
                and ppo_metrics["approx_kl"] <= ppo_config.target_kl
                and ppo_metrics["invalid_delta"] <= 0.0
                and ppo_metrics["truncation_delta"] <= 0.0
            )
        else:
            accepted = False
    else:
        accepted = False
    decision = {
        "method": method,
        "accepted": accepted,
        "changed": changed,
        "gate_mode": gate_mode,
        "old_score": old_score,
        "new_score": new_score,
        "train_old_score": train_old_score,
        "train_new_score": train_new_score,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "train_old_metrics": summarize_slm(train),
        "train_new_metrics": summarize_slm(train_candidate),
    }
    if ppo_metrics is not None and ppo_config is not None:
        decision.update(
            {
                "ppo_config": ppo_config,
                "ppo_metrics": ppo_metrics,
                "ppo_objective": ppo_objective,
            }
        )
    return accepted, decision


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_difficulty_generalization(
    output_dir: Path,
    repetitions: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    methods: list[str] | None = None,
) -> Path:
    methods = methods or DIFFICULTY_METHODS
    unknown_methods = sorted(set(methods) - set(DIFFICULTY_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown difficulty methods: {', '.join(unknown_methods)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "train_envs": DEFAULT_ENVS,
            "test_envs": DIFFICULTY_ENVS,
            "methods": methods,
            "repetitions": repetitions,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "seed": seed,
        },
    )
    write_json(output_dir / "environment_info.json", environment_info())
    rows: list[dict[str, Any]] = []
    all_records: list[EpisodeRecord] = []
    for repetition in range(repetitions):
        for method in methods:
            variables = train_difficulty_variables(
                method,
                repetition,
                train_seeds,
                val_seeds,
                seed,
                output_dir / "base_training",
            )
            records = run_records(
                DIFFICULTY_ENVS,
                method,
                "difficulty_test",
                test_seeds,
                seed + repetition * 10_000 + 50_000,
                variables,
                output_dir / "games" / f"{method}_rep_{repetition:03d}_difficulty_test.jsonl",
            )
            all_records.extend(records)
            rows.append(
                {
                    "suite": "difficulty_generalization",
                    "method": method,
                    "repetition": repetition,
                    **summarize(records),
                }
            )
    per_env_rows = [
        {
            "suite": "difficulty_generalization",
            "method": method,
            "env_id": env_id,
            **summarize([record for record in all_records if record.env_id == env_id and record.variant == method]),
        }
        for method in methods
        for env_id in DIFFICULTY_ENVS
    ]
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "per_env_metrics.csv", per_env_rows)
    write_difficulty_summary(output_dir / "summary.md", rows, per_env_rows, methods)
    return output_dir


def train_difficulty_variables(
    method: str,
    repetition: int,
    train_seeds: int,
    val_seeds: int,
    seed: int,
    output_dir: Path,
) -> dict[str, TextVariable]:
    if method == "fixed_prompt":
        return initial_modular_variables()
    if method == "modular_textgrad":
        _, _, variables = run_baseline_once(
            method=method,
            env_ids=DEFAULT_ENVS,
            repetition=repetition,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=1,
            seed=seed,
            output_dir=output_dir,
        )
        return variables
    if method == "textgrad_rl_plus":
        _, _, variables = run_textgrad_plus_once(
            env_ids=DEFAULT_ENVS,
            repetition=repetition,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=1,
            seed=seed,
            output_dir=output_dir,
            min_mean_delta=0.001,
            max_ci_low_regression=0.0,
        )
        return variables
    if method == "textgrad_policy_iteration":
        _, _, variables = run_policy_iteration_once(
            env_ids=DEFAULT_ENVS,
            repetition=repetition,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=1,
            seed=seed,
            output_dir=output_dir,
            min_mean_delta=0.001,
            max_ci_low_regression=0.0,
        )
        return variables
    if method == "textgrad_ppo":
        _, _, variables = run_textgrad_ppo_once(
            env_ids=DEFAULT_ENVS,
            repetition=repetition,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=1,
            seed=seed,
            output_dir=output_dir,
            min_mean_delta=0.001,
            max_ci_low_regression=0.0,
            ppo_config=TextPPOConfig(),
        )
        return variables
    if method in POLICY_ABLATION_GATE_MODES:
        _, _, variables = run_policy_ablation_once(
            method=method,
            env_ids=DEFAULT_ENVS,
            repetition=repetition,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            test_seeds=1,
            seed=seed,
            output_dir=output_dir,
            min_mean_delta=0.001,
            max_ci_low_regression=0.0,
            ppo_config=TextPPOConfig(),
        )
        return variables
    raise ValueError(f"Unknown difficulty method: {method}")


def write_difficulty_summary(
    path: Path,
    rows: list[dict[str, Any]],
    per_env_rows: list[dict[str, Any]],
    methods: list[str] | None = None,
) -> None:
    methods = methods or DIFFICULTY_METHODS
    lines = [
        "# TextArena Difficulty Generalization",
        "",
        "| method | reward | success | invalid | turns |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        avg = {
            key: sum(row[key] for row in group) / len(group)
            for key in ["average_reward", "success_rate", "invalid_move_rate", "average_turns"]
        }
        lines.append(
            "| {method} | {average_reward:.3f} | {success_rate:.3f} | {invalid_move_rate:.3f} | {average_turns:.3f} |".format(
                method=method,
                **avg,
            )
        )
    lines.extend(
        [
            "",
            "## Per Environment",
            "",
            "| method | env | reward | success | invalid | turns |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_env_rows:
        lines.append(
            "| {method} | {env_id} | {average_reward:.3f} | {success_rate:.3f} | {invalid_move_rate:.3f} | {average_turns:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_slm_suite(
    suite: str,
    env_ids: list[str],
    output_dir: Path,
    base_url: str,
    model_name: str,
    temperature: float,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    timeout: int,
    methods: list[str] | None = None,
) -> Path:
    methods = methods or SLM_METHODS
    unknown_methods = sorted(set(methods) - set(SLM_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown SLM methods: {', '.join(unknown_methods)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    for stale in [
        output_dir / "metrics.csv",
        output_dir / "per_env_metrics.csv",
        output_dir / "summary.md",
        output_dir / "update_decisions.jsonl",
        output_dir / "accepted_updates.json",
        output_dir / "rejected_updates.json",
        output_dir / "gradients.json",
        output_dir / "final_text_variables.json",
    ]:
        if stale.exists():
            stale.unlink()
    for stale in games_dir.glob("*.jsonl"):
        stale.unlink()
    model = OpenAICompatibleChatModel(base_url, model_name, temperature=temperature, max_tokens=64, timeout=timeout)
    write_json(
        output_dir / "config.json",
        {
            "suite": suite,
            "env_ids": env_ids,
            "base_url": base_url,
            "model": model_name,
            "temperature": temperature,
            "methods": methods,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "seed": seed,
        },
    )
    write_json(output_dir / "environment_info.json", environment_info())
    rows: list[dict[str, Any]] = []
    per_method_test_records: dict[str, list[SLMTextArenaRecord]] = {}
    for method_index, method in enumerate(methods):
        method_variables = initial_slm_variables(env_ids)
        method_seed = seed + method_index * 10_000
        if method == "fixed_prompt_slm":
            test = run_slm_records(
                suite,
                env_ids,
                method,
                "test",
                test_seeds,
                seed + 2000,
                model,
                method_variables,
                games_dir / f"{method}_test.jsonl",
            )
            rows.append({"suite": suite, "method": method, "split": "test", "accepted": False, **summarize_slm(test)})
            per_method_test_records[method] = test
            continue

        train = run_slm_records(
            suite,
            env_ids,
            method,
            "train",
            train_seeds,
            method_seed,
            model,
            method_variables,
            games_dir / f"{method}_train.jsonl",
        )
        gradients = slm_gradients_from_records(train)
        write_json(output_dir / f"{method}_gradients.json", gradients)
        candidate = build_slm_candidate(method, method_variables, train, gradients)
        train_candidate = run_slm_records(
            suite,
            env_ids,
            method,
            "train_candidate",
            train_seeds,
            method_seed,
            model,
            candidate,
            games_dir / f"{method}_train_candidate.jsonl",
        )
        val_old = run_slm_records(
            suite,
            env_ids,
            method,
            "val_old",
            val_seeds,
            method_seed + 1000,
            model,
            method_variables,
            games_dir / f"{method}_val_old.jsonl",
        )
        val_candidate = run_slm_records(
            suite,
            env_ids,
            method,
            "val_candidate",
            val_seeds,
            method_seed + 1000,
            model,
            candidate,
            games_dir / f"{method}_val_candidate.jsonl",
        )
        accepted, decision = slm_update_accepted(
            method,
            train,
            train_candidate,
            val_old,
            val_candidate,
            method_variables,
            candidate,
        )
        if accepted:
            method_variables = candidate
        append_jsonl(output_dir / "update_decisions.jsonl", decision)
        write_json(output_dir / f"{method}_final_text_variables.json", method_variables)
        test = run_slm_records(
            suite,
            env_ids,
            method,
            "test",
            test_seeds,
            seed + 2000,
            model,
            method_variables,
            games_dir / f"{method}_test.jsonl",
        )
        rows.extend(
            [
                {"suite": suite, "method": method, "split": "train", "accepted": False, **summarize_slm(train)},
                {
                    "suite": suite,
                    "method": method,
                    "split": "train_candidate",
                    "accepted": accepted,
                    **summarize_slm(train_candidate),
                },
                {"suite": suite, "method": method, "split": "val_old", "accepted": False, **summarize_slm(val_old)},
                {
                    "suite": suite,
                    "method": method,
                    "split": "val_candidate",
                    "accepted": accepted,
                    **summarize_slm(val_candidate),
                },
                {"suite": suite, "method": method, "split": "test", "accepted": accepted, **summarize_slm(test)},
            ]
        )
        per_method_test_records[method] = test
    per_env_rows = [
        {
            "suite": suite,
            "method": method,
            "env_id": env_id,
            **summarize_slm(
                [record for record in per_method_test_records.get(method, []) if record.env_id == env_id]
            ),
        }
        for method in methods
        for env_id in env_ids
    ]
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "per_env_metrics.csv", per_env_rows)
    write_slm_summary(output_dir / "summary.md", suite, model_name, temperature, rows, per_env_rows)
    return output_dir


def write_slm_summary(
    path: Path,
    suite: str,
    model: str,
    temperature: float,
    rows: list[dict[str, Any]],
    per_env_rows: list[dict[str, Any]],
) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    lines = [
        f"# TextArena {suite} SLM Suite",
        "",
        f"- Model: {model}",
        f"- Temperature: {temperature}",
        "",
        "| method | reward | score | success | invalid | truncated | turns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        lines.append(
            "| {method} | {average_reward:.3f} | {average_score:.3f} | {success_rate:.3f} | {invalid_move_rate:.3f} | {truncation_rate:.3f} | {average_turns:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per Environment",
            "",
            "| method | env | reward | score | success | invalid | truncated | turns |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_env_rows:
        lines.append(
            "| {method} | {env_id} | {average_reward:.3f} | {average_score:.3f} | {success_rate:.3f} | {invalid_move_rate:.3f} | {truncation_rate:.3f} | {average_turns:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
    difficulty_methods = [
        method.strip()
        for method in args.difficulty_methods.split(",")
        if method.strip()
    ]
    slm_methods = [
        method.strip()
        for method in args.slm_methods.split(",")
        if method.strip()
    ]
    if "difficulty" in selected:
        run_difficulty_generalization(
            output_dir / "difficulty_generalization",
            repetitions=args.repetitions,
            train_seeds=args.train_seeds,
            val_seeds=args.val_seeds,
            test_seeds=args.test_seeds,
            seed=args.seed,
            methods=difficulty_methods,
        )
    if "puzzle" in selected:
        run_slm_suite(
            "puzzle",
            PUZZLE_SLM_ENVS,
            output_dir / "puzzle_slm",
            args.base_url,
            args.model,
            args.temperature,
            args.slm_train_seeds,
            args.slm_val_seeds,
            args.slm_test_seeds,
            args.seed + 10_000,
            args.timeout,
            slm_methods,
        )
    if "social" in selected:
        run_slm_suite(
            "social",
            SOCIAL_SLM_ENVS,
            output_dir / "social_slm",
            args.base_url,
            args.model,
            args.temperature,
            args.slm_train_seeds,
            args.slm_val_seeds,
            args.slm_test_seeds,
            args.seed + 20_000,
            args.timeout,
            slm_methods,
        )
    if "real_slm" in selected:
        run_slm_suite(
            "real_slm",
            REAL_SLM_ENVS,
            output_dir / "real_slm",
            args.base_url,
            args.model,
            args.temperature,
            args.slm_train_seeds,
            args.slm_val_seeds,
            args.slm_test_seeds,
            args.seed + 30_000,
            args.timeout,
            slm_methods,
        )
    write_index(output_dir, selected)
    return output_dir


def write_index(output_dir: Path, selected: set[str]) -> None:
    lines = ["# Expanded TextArena Benchmark Suites", ""]
    for suite, rel in [
        ("difficulty", "difficulty_generalization/summary.md"),
        ("puzzle", "puzzle_slm/summary.md"),
        ("social", "social_slm/summary.md"),
        ("real_slm", "real_slm/summary.md"),
    ]:
        if suite in selected:
            lines.append(f"- {suite}: `{rel}`")
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run expanded TextArena benchmark suites.")
    parser.add_argument("--suites", default="difficulty,puzzle,social,real_slm")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--slm-train-seeds", type=int, default=1)
    parser.add_argument("--slm-val-seeds", type=int, default=1)
    parser.add_argument("--slm-test-seeds", type=int, default=1)
    parser.add_argument("--difficulty-methods", default=",".join(DIFFICULTY_METHODS))
    parser.add_argument("--slm-methods", default=",".join(SLM_METHODS))
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--base-url", default=os.getenv("TEXTGRAD_RL_LLM_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("TEXTGRAD_RL_LLM_MODEL", "qwen2.5:3b"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEXTGRAD_RL_LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", default="runs/textarena_expanded_suites")
    return parser


def main() -> None:
    output_dir = run_all(build_parser().parse_args())
    print(f"Expanded TextArena benchmark artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
