"""TextGrad policy iteration with action advantages and a value critic."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_multienv_compare import (
    canonical_env_id,
    DEFAULT_ENVS,
    EpisodeRecord,
    rule_for_env,
    run_records,
    summarize,
)
from textgrad_rl.benchmarks.textarena_paper_suite import initial_modular_variables, slug_env
from textgrad_rl.benchmarks.textarena_textgrad_plus import (
    CandidateEdit,
    apply_rule_library,
    bootstrap_ci,
    bootstrap_delta_ci,
    build_per_env_rows,
    gate_accepts,
    generate_candidate_edits,
    record_score,
    run_baseline_once,
    run_textgrad_plus_once,
    variables_changed,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


METHODS = [
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

POLICY_ABLATION_GATE_MODES = {
    "textgrad_rl_no_gate": "no_gate",
    "textgrad_rl_train_val": "train_val",
    "textgrad_rl_kl_gate": "kl_gate",
    "textgrad_rl_clipped_surrogate": "clipped_surrogate",
    "textgrad_rl_ppo": "ppo",
}

EFFICIENT_TURN_TARGETS = {
    "Bandit-v0": 21,
    "Blackjack-v0": 5,
    "ConnectFour-v0": 19,
    "FrozenLake-v0": 6,
    "GuessTheNumber-v0": 5,
    "LightsOut-v0": 8,
    "Mastermind-v0": 5,
    "Nim-v0": 12,
    "ReverseTicTacToe-v0": 9,
    "TowerOfHanoi-v0": 7,
}


@dataclass
class ActionCredit:
    env_id: str
    seed: int
    target_side: int | None
    step_index: int
    action: str
    return_score: float
    baseline_score: float
    advantage: float
    credit_label: str
    evidence: str


@dataclass
class AdvantageAssignment:
    env_id: str
    target_variable_name: str
    failure_count: int
    invalid_count: int
    average_reward: float
    mean_negative_advantage: float
    worst_action: str
    causal_factor: str
    evidence: str


@dataclass
class TextPPOConfig:
    """Trust-region settings for PPO-style text-policy updates."""

    clip_epsilon: float = 0.2
    target_kl: float = 0.2
    score_scale: float = 5.0
    min_surrogate_delta: float = 0.001


@dataclass
class TextArenaValueCritic:
    """Tabular value critic over TextArena environment returns."""

    env_baselines: dict[str, float] = field(default_factory=dict)
    global_baseline: float = 0.0
    candidate_values: dict[str, float] = field(default_factory=dict)

    def fit(self, records: list[EpisodeRecord]) -> None:
        scores = [record_score(record) for record in records]
        self.global_baseline = sum(scores) / len(scores) if scores else 0.0
        for env_id in sorted({record.env_id for record in records}):
            env_scores = [record_score(record) for record in records if record.env_id == env_id]
            self.env_baselines[env_id] = sum(env_scores) / len(env_scores) if env_scores else self.global_baseline

    def baseline(self, env_id: str) -> float:
        return self.env_baselines.get(env_id, self.global_baseline)

    def estimate_advantage(self, records: list[EpisodeRecord]) -> float:
        if not records:
            return 0.0
        advantages = [record_score(record) - self.baseline(record.env_id) for record in records]
        return sum(advantages) / len(advantages)

    def remember_candidate(self, name: str, records: list[EpisodeRecord]) -> None:
        self.candidate_values[name] = self.estimate_advantage(records)


class ReplayBuffer:
    """Small replay store for policy-iteration diagnostics and gating."""

    def __init__(self) -> None:
        self.records: dict[str, list[EpisodeRecord]] = {}
        self.action_credits: list[ActionCredit] = []
        self.candidate_evaluations: list[dict[str, Any]] = []

    def add_records(self, split: str, records: list[EpisodeRecord]) -> None:
        self.records.setdefault(split, []).extend(records)

    def add_action_credits(self, credits: list[ActionCredit]) -> None:
        self.action_credits.extend(credits)

    def add_candidate_evaluation(self, details: dict[str, Any]) -> None:
        self.candidate_evaluations.append(details)

    def write(self, path: Path) -> None:
        write_json(
            path,
            {
                "record_counts": {split: len(records) for split, records in self.records.items()},
                "action_credits": self.action_credits,
                "candidate_evaluations": self.candidate_evaluations,
            },
        )


def assign_action_credits(records: list[EpisodeRecord], critic: TextArenaValueCritic) -> list[ActionCredit]:
    """Assign action-level advantages from final returns and trajectory structure."""

    credits: list[ActionCredit] = []
    for record in records:
        final_score = record_score(record)
        baseline = critic.baseline(record.env_id)
        efficient_turns = EFFICIENT_TURN_TARGETS.get(canonical_env_id(record.env_id), max(1, record.turns))
        seen_actions: set[str] = set()
        for step_index, action in enumerate(record.actions):
            advantage = final_score - baseline
            labels = []
            if action in seen_actions:
                advantage -= 0.25
                labels.append("repeated_action")
            seen_actions.add(action)
            if record.invalid_move and step_index == len(record.actions) - 1:
                advantage -= 0.5
                labels.append("terminal_invalid")
            if step_index + 1 > efficient_turns and not record.success:
                advantage -= 0.05 * (step_index + 1 - efficient_turns)
                labels.append("inefficient_tail")
            if record.success and step_index == len(record.actions) - 1:
                advantage += 0.2
                labels.append("terminal_success")
            if record.success and record.turns <= efficient_turns:
                advantage += 0.1
                labels.append("efficient_success_step")
            if not labels:
                labels.append("trajectory_return")
            credits.append(
                ActionCredit(
                    env_id=record.env_id,
                    seed=record.seed,
                    target_side=record.target_side,
                    step_index=step_index,
                    action=action,
                    return_score=final_score,
                    baseline_score=baseline,
                    advantage=advantage,
                    credit_label=",".join(labels),
                    evidence=(
                        f"{record.env_id} seed={record.seed} step={step_index} action={action} "
                        f"advantage={advantage:.3f} final_reward={record.reward:.3f}"
                    ),
                )
            )
    return credits


def advantage_assignments_from_records(
    records: list[EpisodeRecord],
    credits: list[ActionCredit],
) -> list[AdvantageAssignment]:
    assignments: list[AdvantageAssignment] = []
    for env_id in sorted({record.env_id for record in records}):
        group = [record for record in records if record.env_id == env_id]
        failures = [record for record in group if record.reward < 1.0 or record.invalid_move]
        env_credits = [credit for credit in credits if credit.env_id == env_id]
        negative = [credit for credit in env_credits if credit.advantage < 0.0]
        if not failures and not negative:
            continue
        invalid_count = sum(record.invalid_move for record in failures)
        worst = min(env_credits, key=lambda credit: credit.advantage) if env_credits else None
        labels = ",".join(sorted({label for credit in negative for label in credit.credit_label.split(",")}))
        if invalid_count:
            factor = "invalid_action"
        elif "inefficient_tail" in labels:
            factor = "long_horizon_credit"
        elif "repeated_action" in labels:
            factor = "repeated_action"
        else:
            factor = "environment_strategy"
        mean_negative = sum(credit.advantage for credit in negative) / len(negative) if negative else 0.0
        assignments.append(
            AdvantageAssignment(
                env_id=env_id,
                target_variable_name=f"{slug_env(canonical_env_id(env_id))}_strategy_prompt",
                failure_count=len(failures),
                invalid_count=invalid_count,
                average_reward=sum(record.reward for record in group) / len(group),
                mean_negative_advantage=mean_negative,
                worst_action=worst.action if worst else "",
                causal_factor=factor,
                evidence=(
                    f"{len(failures)}/{len(group)} episodes failed or had invalid moves. "
                    f"mean_negative_advantage={mean_negative:.3f}. "
                    f"worst_action={worst.action if worst else '<none>'}. "
                    f"worst_evidence={worst.evidence if worst else '<none>'}."
                ),
            )
        )
    return sorted(assignments, key=lambda item: (item.mean_negative_advantage, item.env_id))


def make_advantage_gradient(assignment: AdvantageAssignment, rule: str) -> TextualGradient:
    confidence = min(0.98, max(0.55, 0.65 + abs(assignment.mean_negative_advantage) / 2.0))
    return TextualGradient(
        target_variable_name=assignment.target_variable_name,
        failure_mode=f"{assignment.env_id} negative-advantage {assignment.causal_factor}",
        evidence_from_trajectory=assignment.evidence,
        gradient_text=(
            "Increase probability of actions consistent with the rule and decrease actions "
            f"like {assignment.worst_action!r}, which received negative advantage."
        ),
        suggested_edit=f"Add a rule: {rule}",
        confidence=confidence,
        forbidden_shortcuts=["hidden state", "environment mutation", "invalid action formats"],
    )


def generate_policy_candidates(
    assignment: AdvantageAssignment,
    variables: dict[str, TextVariable],
    optimizer: TextualGradientDescent,
) -> list[CandidateEdit]:
    """Population search over text-policy candidates for one assignment."""

    env_rule = rule_for_env(assignment.env_id)
    candidates: list[CandidateEdit] = []
    weighted_gradient = make_advantage_gradient(assignment, env_rule)
    weighted = optimizer.step(
        variables,
        [weighted_gradient],
        constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
    )
    candidates.append(
        CandidateEdit(
            name="advantage_weighted_env_rule",
            env_id=assignment.env_id,
            target_variable_name=assignment.target_variable_name,
            rule=env_rule,
            variables=weighted,
            gradients=[weighted_gradient],
        )
    )

    from textgrad_rl.benchmarks.textarena_textgrad_plus import CausalAssignment

    causal = CausalAssignment(
        env_id=assignment.env_id,
        target_variable_name=assignment.target_variable_name,
        failure_count=assignment.failure_count,
        invalid_count=assignment.invalid_count,
        average_reward=assignment.average_reward,
        causal_factor=assignment.causal_factor,
        evidence=assignment.evidence,
    )
    candidates.extend(generate_candidate_edits(causal, variables, optimizer))
    return candidates


def policy_objective(details: dict[str, Any], critic_advantage: float, assignment: AdvantageAssignment) -> float:
    return (
        float(details["mean_delta"])
        + 0.25 * critic_advantage
        + 0.05 * abs(assignment.mean_negative_advantage)
        - 0.01 * float(details["new_metrics"]["average_turns"])
    )


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def text_behavior_ratio(old_score: float, new_score: float, advantage: float, score_scale: float) -> float:
    """Estimate a PPO ratio for text policies from paired old/new rollouts.

    We do not have token log-probs for prompt text variables. Instead, this proxy
    treats an improvement on a positive-advantage old trajectory as increasing
    the tendency to keep that behavior, and an improvement on a negative-
    advantage old trajectory as decreasing the tendency to repeat it.
    """

    if abs(advantage) < 1e-12:
        return 1.0
    scaled_delta = (new_score - old_score) / max(score_scale, 1e-9)
    movement = math.tanh(scaled_delta)
    ratio = 1.0 + movement if advantage > 0.0 else 1.0 - movement
    return max(1e-6, ratio)


def clipped_text_surrogate(advantage: float, ratio: float, clip_epsilon: float) -> float:
    clipped_ratio = _clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)


def text_ppo_metrics(
    old_records: list[EpisodeRecord],
    new_records: list[EpisodeRecord],
    critic: TextArenaValueCritic,
    config: TextPPOConfig,
    fallback_advantage: float = 0.0,
) -> dict[str, Any]:
    ratios: list[float] = []
    old_surrogates: list[float] = []
    new_surrogates: list[float] = []
    score_deltas: list[float] = []
    action_changes: list[float] = []
    paired = list(zip(old_records, new_records))
    for old, new in paired:
        old_score = record_score(old)
        new_score = record_score(new)
        advantage = old_score - critic.baseline(old.env_id)
        if abs(advantage) < 1e-12 and fallback_advantage:
            advantage = fallback_advantage
        ratio = text_behavior_ratio(old_score, new_score, advantage, config.score_scale)
        ratios.append(ratio)
        old_surrogates.append(advantage)
        new_surrogates.append(clipped_text_surrogate(advantage, ratio, config.clip_epsilon))
        score_deltas.append(new_score - old_score)
        action_changes.append(0.0 if old.actions == new.actions else 1.0)

    n = len(paired)
    old_metrics = summarize(old_records)
    new_metrics = summarize(new_records)
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
            "fallback_advantage": fallback_advantage,
            "invalid_delta": new_metrics["invalid_move_rate"] - old_metrics["invalid_move_rate"],
            "turn_delta": new_metrics["average_turns"] - old_metrics["average_turns"],
        }
    old_surrogate = sum(old_surrogates) / n
    clipped_surrogate = sum(new_surrogates) / n
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
        "fallback_advantage": fallback_advantage,
        "invalid_delta": new_metrics["invalid_move_rate"] - old_metrics["invalid_move_rate"],
        "turn_delta": new_metrics["average_turns"] - old_metrics["average_turns"],
    }


def ppo_text_objective(details: dict[str, Any]) -> float:
    return (
        float(details["surrogate_delta"])
        + 0.25 * float(details["mean_delta"])
        - 0.5 * float(details["approx_kl"])
        - max(0.0, float(details["invalid_delta"]))
        - 0.005 * max(0.0, float(details["turn_delta"]))
    )


def ppo_gate_accepts(
    old_records: list[EpisodeRecord],
    new_records: list[EpisodeRecord],
    critic: TextArenaValueCritic,
    config: TextPPOConfig,
    seed: int,
    min_mean_delta: float,
    max_ci_low_regression: float,
    fallback_advantage: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    bootstrap_accepted, details = gate_accepts(
        old_records,
        new_records,
        seed=seed,
        min_mean_delta=min_mean_delta,
        max_ci_low_regression=max_ci_low_regression,
    )
    ppo_metrics = text_ppo_metrics(old_records, new_records, critic, config, fallback_advantage)
    details.update(ppo_metrics)
    accepted = (
        bootstrap_accepted
        and ppo_metrics["surrogate_delta"] > config.min_surrogate_delta
        and ppo_metrics["approx_kl"] <= config.target_kl
        and ppo_metrics["invalid_delta"] <= 0.0
    )
    details.update(
        {
            "accepted": accepted,
            "bootstrap_accepted": bootstrap_accepted,
            "ppo_clip_epsilon": config.clip_epsilon,
            "ppo_target_kl": config.target_kl,
            "ppo_score_scale": config.score_scale,
            "ppo_min_surrogate_delta": config.min_surrogate_delta,
            "ppo_objective": ppo_text_objective(details),
        }
    )
    return accepted, details


def policy_ablation_gate_accepts(
    gate_mode: str,
    old_records: list[EpisodeRecord],
    new_records: list[EpisodeRecord],
    critic: TextArenaValueCritic,
    config: TextPPOConfig,
    seed: int,
    min_mean_delta: float,
    max_ci_low_regression: float,
    fallback_advantage: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Accept a TextGrad policy candidate under one ablation gate."""

    bootstrap_accepted, details = gate_accepts(
        old_records,
        new_records,
        seed=seed,
        min_mean_delta=min_mean_delta,
        max_ci_low_regression=max_ci_low_regression,
    )
    ppo_metrics = text_ppo_metrics(old_records, new_records, critic, config, fallback_advantage)
    details.update(ppo_metrics)
    if gate_mode == "no_gate":
        accepted = True
    elif gate_mode == "train_val":
        accepted = bootstrap_accepted
    elif gate_mode == "kl_gate":
        accepted = details["mean_delta"] > min_mean_delta and ppo_metrics["approx_kl"] <= config.target_kl
    elif gate_mode == "clipped_surrogate":
        accepted = ppo_metrics["surrogate_delta"] > config.min_surrogate_delta
    elif gate_mode == "ppo":
        accepted = (
            bootstrap_accepted
            and ppo_metrics["surrogate_delta"] > config.min_surrogate_delta
            and ppo_metrics["approx_kl"] <= config.target_kl
            and ppo_metrics["invalid_delta"] <= 0.0
        )
    else:
        raise ValueError(f"Unknown policy ablation gate mode: {gate_mode}")
    details.update(
        {
            "accepted": accepted,
            "bootstrap_accepted": bootstrap_accepted,
            "gate_mode": gate_mode,
            "ppo_clip_epsilon": config.clip_epsilon,
            "ppo_target_kl": config.target_kl,
            "ppo_score_scale": config.score_scale,
            "ppo_min_surrogate_delta": config.min_surrogate_delta,
            "ppo_objective": ppo_text_objective(details),
        }
    )
    return accepted, details


def run_policy_ablation_once(
    method: str,
    env_ids: list[str],
    repetition: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    min_mean_delta: float,
    max_ci_low_regression: float,
    ppo_config: TextPPOConfig,
) -> tuple[list[dict[str, Any]], list[EpisodeRecord], dict[str, TextVariable]]:
    gate_mode = POLICY_ABLATION_GATE_MODES[method]
    method_dir = output_dir / "methods" / method / f"rep_{repetition:03d}"
    games_dir = method_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    variables = initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    base_seed = seed + repetition * 10_000
    replay = ReplayBuffer()
    critic = TextArenaValueCritic()
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    library: dict[str, Any] = {"accepted_rules": [], "rejected_rules": [], "gate_mode": gate_mode}

    train_records = run_records(
        env_ids,
        method,
        "train",
        train_seeds,
        base_seed,
        variables,
        games_dir / "train.jsonl",
    )
    replay.add_records("train", train_records)
    critic.fit(train_records)
    action_credits = assign_action_credits(train_records, critic)
    replay.add_action_credits(action_credits)
    assignments = advantage_assignments_from_records(train_records, action_credits)
    rows.append(row_for(method, repetition, "train", train_records))
    write_json(method_dir / "ppo_config.json", ppo_config)
    write_json(method_dir / "action_credits.json", action_credits)
    write_json(method_dir / "advantage_assignments.json", assignments)
    write_json(method_dir / "value_critic.json", critic)

    for assignment_index, assignment in enumerate(assignments):
        best_candidate: CandidateEdit | None = None
        best_details: dict[str, Any] | None = None
        best_objective = float("-inf")
        candidates = generate_policy_candidates(assignment, variables, optimizer)
        for candidate_index, candidate in enumerate(candidates):
            candidate_count += 1
            candidate_slug = f"{assignment_index:02d}_{slug_env(candidate.env_id)}_{candidate.name}_{candidate_index:02d}"
            old_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_replay",
                train_seeds,
                base_seed,
                variables,
                games_dir / f"{candidate_slug}_old_replay.jsonl",
            )
            new_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_replay",
                train_seeds,
                base_seed,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_replay.jsonl",
            )
            old_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_val",
                val_seeds,
                base_seed + 1000,
                variables,
                games_dir / f"{candidate_slug}_old_val.jsonl",
            )
            new_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_val",
                val_seeds,
                base_seed + 1000,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_val.jsonl",
            )
            replay.add_records(f"{candidate_slug}_old_replay", old_replay)
            replay.add_records(f"{candidate_slug}_new_replay", new_replay)
            replay.add_records(f"{candidate_slug}_old_val", old_val)
            replay.add_records(f"{candidate_slug}_new_val", new_val)
            critic_advantage = critic.estimate_advantage(new_replay + new_val)
            critic.remember_candidate(candidate_slug, new_replay + new_val)
            accepted, details = policy_ablation_gate_accepts(
                gate_mode,
                old_replay + old_val,
                new_replay + new_val,
                critic,
                ppo_config,
                seed=base_seed + assignment_index * 101 + candidate_index,
                min_mean_delta=min_mean_delta,
                max_ci_low_regression=max_ci_low_regression,
                fallback_advantage=assignment.mean_negative_advantage,
            )
            changed = variables_changed(variables, candidate.variables)
            policy_score = policy_objective(details, critic_advantage, assignment)
            selection_objective = details["ppo_objective"] if gate_mode in {"clipped_surrogate", "ppo"} else policy_score
            details.update(
                {
                    "candidate_name": candidate.name,
                    "candidate_slug": candidate_slug,
                    "env_id": candidate.env_id,
                    "target_variable_name": candidate.target_variable_name,
                    "rule": candidate.rule,
                    "changed": changed,
                    "critic_advantage": critic_advantage,
                    "policy_objective": policy_score,
                    "selection_objective": selection_objective,
                    "assignment_mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            replay.add_candidate_evaluation(details)
            append_jsonl(method_dir / "candidate_evaluations.jsonl", details)
            if accepted and changed and selection_objective > best_objective:
                best_candidate = candidate
                best_details = details
                best_objective = selection_objective

        if best_candidate is None or best_details is None:
            rejected_count += len(candidates)
            library["rejected_rules"].append(
                {
                    "env_id": assignment.env_id,
                    "target_variable_name": assignment.target_variable_name,
                    "reason": f"no candidate passed {gate_mode} gate",
                    "mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            continue

        variables = best_candidate.variables
        accepted_count += 1
        rejected_count += len(candidates) - 1
        library["accepted_rules"].append(
            {
                "env_id": best_candidate.env_id,
                "target_variable_name": best_candidate.target_variable_name,
                "candidate_name": best_candidate.name,
                "rule": best_candidate.rule,
                "gate_mode": gate_mode,
                "mean_delta": best_details["mean_delta"],
                "ci_low": best_details["ci_low"],
                "ci_high": best_details["ci_high"],
                "surrogate_delta": best_details["surrogate_delta"],
                "approx_kl": best_details["approx_kl"],
                "clip_fraction": best_details["clip_fraction"],
                "ppo_objective": best_details["ppo_objective"],
                "selection_objective": best_details["selection_objective"],
                "worst_action": best_details["worst_action"],
            }
        )
        append_jsonl(method_dir / "accepted_updates.jsonl", best_details)

    library_variables = apply_rule_library(library)
    write_json(method_dir / "learned_rule_library.json", library)
    write_json(method_dir / "final_text_variables.json", library_variables)
    replay.write(method_dir / "replay_buffer.json")
    test_records = run_records(
        env_ids,
        method,
        "test",
        test_seeds,
        base_seed + 2000,
        library_variables,
        games_dir / "test.jsonl",
    )
    rows.append(
        row_for(
            method,
            repetition,
            "test",
            test_records,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            candidate_count=candidate_count,
        )
    )
    return rows, test_records, library_variables


def run_policy_iteration_once(
    env_ids: list[str],
    repetition: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    min_mean_delta: float,
    max_ci_low_regression: float,
) -> tuple[list[dict[str, Any]], list[EpisodeRecord], dict[str, TextVariable]]:
    method = "textgrad_policy_iteration"
    method_dir = output_dir / "methods" / method / f"rep_{repetition:03d}"
    games_dir = method_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    variables = initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    base_seed = seed + repetition * 10_000
    replay = ReplayBuffer()
    critic = TextArenaValueCritic()
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    library: dict[str, Any] = {"accepted_rules": [], "rejected_rules": []}

    train_records = run_records(
        env_ids,
        method,
        "train",
        train_seeds,
        base_seed,
        variables,
        games_dir / "train.jsonl",
    )
    replay.add_records("train", train_records)
    critic.fit(train_records)
    action_credits = assign_action_credits(train_records, critic)
    replay.add_action_credits(action_credits)
    assignments = advantage_assignments_from_records(train_records, action_credits)
    rows.append(row_for(method, repetition, "train", train_records))
    write_json(method_dir / "action_credits.json", action_credits)
    write_json(method_dir / "advantage_assignments.json", assignments)
    write_json(method_dir / "value_critic.json", critic)

    for assignment_index, assignment in enumerate(assignments):
        best_candidate: CandidateEdit | None = None
        best_details: dict[str, Any] | None = None
        best_objective = float("-inf")
        candidates = generate_policy_candidates(assignment, variables, optimizer)
        for candidate_index, candidate in enumerate(candidates):
            candidate_count += 1
            candidate_slug = f"{assignment_index:02d}_{slug_env(candidate.env_id)}_{candidate.name}_{candidate_index:02d}"
            old_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_replay",
                train_seeds,
                base_seed,
                variables,
                games_dir / f"{candidate_slug}_old_replay.jsonl",
            )
            new_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_replay",
                train_seeds,
                base_seed,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_replay.jsonl",
            )
            old_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_val",
                val_seeds,
                base_seed + 1000,
                variables,
                games_dir / f"{candidate_slug}_old_val.jsonl",
            )
            new_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_val",
                val_seeds,
                base_seed + 1000,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_val.jsonl",
            )
            replay.add_records(f"{candidate_slug}_old_replay", old_replay)
            replay.add_records(f"{candidate_slug}_new_replay", new_replay)
            replay.add_records(f"{candidate_slug}_old_val", old_val)
            replay.add_records(f"{candidate_slug}_new_val", new_val)
            critic_advantage = critic.estimate_advantage(new_replay + new_val)
            critic.remember_candidate(candidate_slug, new_replay + new_val)
            accepted, details = gate_accepts(
                old_replay + old_val,
                new_replay + new_val,
                seed=base_seed + assignment_index * 101 + candidate_index,
                min_mean_delta=min_mean_delta,
                max_ci_low_regression=max_ci_low_regression,
            )
            changed = variables_changed(variables, candidate.variables)
            objective = policy_objective(details, critic_advantage, assignment)
            details.update(
                {
                    "candidate_name": candidate.name,
                    "candidate_slug": candidate_slug,
                    "env_id": candidate.env_id,
                    "target_variable_name": candidate.target_variable_name,
                    "rule": candidate.rule,
                    "changed": changed,
                    "critic_advantage": critic_advantage,
                    "policy_objective": objective,
                    "assignment_mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            replay.add_candidate_evaluation(details)
            append_jsonl(method_dir / "candidate_evaluations.jsonl", details)
            if accepted and changed and objective > best_objective:
                best_candidate = candidate
                best_details = details
                best_objective = objective

        if best_candidate is None or best_details is None:
            rejected_count += len(candidates)
            library["rejected_rules"].append(
                {
                    "env_id": assignment.env_id,
                    "target_variable_name": assignment.target_variable_name,
                    "reason": "no candidate passed advantage replay-bootstrap gate",
                    "mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            continue

        variables = best_candidate.variables
        accepted_count += 1
        rejected_count += len(candidates) - 1
        library["accepted_rules"].append(
            {
                "env_id": best_candidate.env_id,
                "target_variable_name": best_candidate.target_variable_name,
                "candidate_name": best_candidate.name,
                "rule": best_candidate.rule,
                "mean_delta": best_details["mean_delta"],
                "ci_low": best_details["ci_low"],
                "ci_high": best_details["ci_high"],
                "critic_advantage": best_details["critic_advantage"],
                "policy_objective": best_details["policy_objective"],
                "worst_action": best_details["worst_action"],
            }
        )
        append_jsonl(method_dir / "accepted_updates.jsonl", best_details)

    library_variables = apply_rule_library(library)
    write_json(method_dir / "learned_rule_library.json", library)
    write_json(method_dir / "final_text_variables.json", library_variables)
    replay.write(method_dir / "replay_buffer.json")
    test_records = run_records(
        env_ids,
        method,
        "test",
        test_seeds,
        base_seed + 2000,
        library_variables,
        games_dir / "test.jsonl",
    )
    rows.append(
        row_for(
            method,
            repetition,
            "test",
            test_records,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            candidate_count=candidate_count,
        )
    )
    return rows, test_records, library_variables


def run_textgrad_ppo_once(
    env_ids: list[str],
    repetition: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    min_mean_delta: float,
    max_ci_low_regression: float,
    ppo_config: TextPPOConfig,
) -> tuple[list[dict[str, Any]], list[EpisodeRecord], dict[str, TextVariable]]:
    method = "textgrad_ppo"
    method_dir = output_dir / "methods" / method / f"rep_{repetition:03d}"
    games_dir = method_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    variables = initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    base_seed = seed + repetition * 10_000
    replay = ReplayBuffer()
    critic = TextArenaValueCritic()
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    library: dict[str, Any] = {"accepted_rules": [], "rejected_rules": []}

    train_records = run_records(
        env_ids,
        method,
        "train",
        train_seeds,
        base_seed,
        variables,
        games_dir / "train.jsonl",
    )
    replay.add_records("train", train_records)
    critic.fit(train_records)
    action_credits = assign_action_credits(train_records, critic)
    replay.add_action_credits(action_credits)
    assignments = advantage_assignments_from_records(train_records, action_credits)
    rows.append(row_for(method, repetition, "train", train_records))
    write_json(method_dir / "ppo_config.json", ppo_config)
    write_json(method_dir / "action_credits.json", action_credits)
    write_json(method_dir / "advantage_assignments.json", assignments)
    write_json(method_dir / "value_critic.json", critic)

    for assignment_index, assignment in enumerate(assignments):
        best_candidate: CandidateEdit | None = None
        best_details: dict[str, Any] | None = None
        best_objective = float("-inf")
        candidates = generate_policy_candidates(assignment, variables, optimizer)
        for candidate_index, candidate in enumerate(candidates):
            candidate_count += 1
            candidate_slug = f"{assignment_index:02d}_{slug_env(candidate.env_id)}_{candidate.name}_{candidate_index:02d}"
            old_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_replay",
                train_seeds,
                base_seed,
                variables,
                games_dir / f"{candidate_slug}_old_replay.jsonl",
            )
            new_replay = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_replay",
                train_seeds,
                base_seed,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_replay.jsonl",
            )
            old_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_old_val",
                val_seeds,
                base_seed + 1000,
                variables,
                games_dir / f"{candidate_slug}_old_val.jsonl",
            )
            new_val = run_records(
                [candidate.env_id],
                method,
                f"{candidate_slug}_new_val",
                val_seeds,
                base_seed + 1000,
                candidate.variables,
                games_dir / f"{candidate_slug}_new_val.jsonl",
            )
            replay.add_records(f"{candidate_slug}_old_replay", old_replay)
            replay.add_records(f"{candidate_slug}_new_replay", new_replay)
            replay.add_records(f"{candidate_slug}_old_val", old_val)
            replay.add_records(f"{candidate_slug}_new_val", new_val)
            accepted, details = ppo_gate_accepts(
                old_replay + old_val,
                new_replay + new_val,
                critic,
                ppo_config,
                seed=base_seed + assignment_index * 101 + candidate_index,
                min_mean_delta=min_mean_delta,
                max_ci_low_regression=max_ci_low_regression,
                fallback_advantage=assignment.mean_negative_advantage,
            )
            changed = variables_changed(variables, candidate.variables)
            details.update(
                {
                    "candidate_name": candidate.name,
                    "candidate_slug": candidate_slug,
                    "env_id": candidate.env_id,
                    "target_variable_name": candidate.target_variable_name,
                    "rule": candidate.rule,
                    "changed": changed,
                    "assignment_mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            replay.add_candidate_evaluation(details)
            append_jsonl(method_dir / "ppo_candidate_evaluations.jsonl", details)
            if accepted and changed and details["ppo_objective"] > best_objective:
                best_candidate = candidate
                best_details = details
                best_objective = details["ppo_objective"]

        if best_candidate is None or best_details is None:
            rejected_count += len(candidates)
            library["rejected_rules"].append(
                {
                    "env_id": assignment.env_id,
                    "target_variable_name": assignment.target_variable_name,
                    "reason": "no candidate passed clipped text-PPO trust-region gate",
                    "mean_negative_advantage": assignment.mean_negative_advantage,
                    "worst_action": assignment.worst_action,
                }
            )
            continue

        variables = best_candidate.variables
        accepted_count += 1
        rejected_count += len(candidates) - 1
        library["accepted_rules"].append(
            {
                "env_id": best_candidate.env_id,
                "target_variable_name": best_candidate.target_variable_name,
                "candidate_name": best_candidate.name,
                "rule": best_candidate.rule,
                "mean_delta": best_details["mean_delta"],
                "ci_low": best_details["ci_low"],
                "ci_high": best_details["ci_high"],
                "surrogate_delta": best_details["surrogate_delta"],
                "approx_kl": best_details["approx_kl"],
                "clip_fraction": best_details["clip_fraction"],
                "ppo_objective": best_details["ppo_objective"],
                "worst_action": best_details["worst_action"],
            }
        )
        append_jsonl(method_dir / "accepted_updates.jsonl", best_details)

    library_variables = apply_rule_library(library)
    write_json(method_dir / "learned_rule_library.json", library)
    write_json(method_dir / "final_text_variables.json", library_variables)
    replay.write(method_dir / "replay_buffer.json")
    test_records = run_records(
        env_ids,
        method,
        "test",
        test_seeds,
        base_seed + 2000,
        library_variables,
        games_dir / "test.jsonl",
    )
    rows.append(
        row_for(
            method,
            repetition,
            "test",
            test_records,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            candidate_count=candidate_count,
        )
    )
    return rows, test_records, library_variables


def row_for(
    method: str,
    repetition: int,
    split: str,
    records: list[EpisodeRecord],
    accepted_count: int = 0,
    rejected_count: int = 0,
    candidate_count: int = 0,
) -> dict[str, Any]:
    return {
        "method": method,
        "repetition": repetition,
        "split": split,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "candidate_count": candidate_count,
        **summarize(records),
    }


def build_ci_rows(records_by_method: dict[str, list[EpisodeRecord]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, records in sorted(records_by_method.items()):
        for metric, values in {
            "reward": [record.reward for record in records],
            "success": [1.0 if record.success else 0.0 for record in records],
            "invalid_move": [1.0 if record.invalid_move else 0.0 for record in records],
            "turns": [float(record.turns) for record in records],
        }.items():
            mean, lo, hi = bootstrap_ci(values, seed + len(method) + len(metric))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(values),
                }
            )
    return rows


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
    env_ids: list[str],
    metrics_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
    methods: list[str] | None = None,
) -> None:
    methods = methods or METHODS
    test_rows = [row for row in metrics_rows if row["split"] == "test"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        grouped.setdefault(row["method"], []).append(row)
    lines = [
        "# TextGrad Policy Iteration Suite",
        "",
        f"- Environments: {len(env_ids)}",
        f"- Methods: {', '.join(methods)}",
        "",
        "## RL Strengthening",
        "",
        "- Action-level credit assignment from trajectory actions, invalid moves, repeated actions, turn efficiency, and returns.",
        "- Advantage-weighted textual gradients targeted at the worst negative-advantage actions.",
        "- Candidate prompt policy search over a text-policy population.",
        "- Replay buffer containing train, replay, validation, action-credit, and candidate-evaluation artifacts.",
        "- Tabular value critic estimating environment baselines and candidate advantages.",
        "- Text-PPO variant with clipped behavioral ratios, surrogate-delta acceptance, and KL-style trust-region limits.",
        "",
        "## Test Means Across Repetitions",
        "",
        "| method | reward | success | invalid | turns | accepted | candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        rows = grouped.get(method, [])
        if not rows:
            continue
        lines.append(
            "| {method} | {reward:.3f} | {success:.3f} | {invalid:.3f} | {turns:.3f} | {accepted:.1f} | {candidates:.1f} |".format(
                method=method,
                reward=sum(row["average_reward"] for row in rows) / len(rows),
                success=sum(row["success_rate"] for row in rows) / len(rows),
                invalid=sum(row["invalid_move_rate"] for row in rows) / len(rows),
                turns=sum(row["average_turns"] for row in rows) / len(rows),
                accepted=sum(row["accepted_count"] for row in rows) / len(rows),
                candidates=sum(row["candidate_count"] for row in rows) / len(rows),
            )
        )
    lines.extend(["", "## Bootstrap 95% CIs", "", "| method | metric | mean | ci_low | ci_high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in ci_rows:
        if row["metric"] in {"reward", "success", "invalid_move"}:
            lines.append(
                "| {method} | {metric} | {mean:.3f} | {ci_low:.3f} | {ci_high:.3f} | {n} |".format(**row)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_policy_iteration_suite(
    env_ids: list[str],
    repetitions: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    min_mean_delta: float,
    max_ci_low_regression: float,
    ppo_config: TextPPOConfig | None = None,
    methods: list[str] | None = None,
) -> Path:
    try:
        import textarena as ta
        import textarena.api as ta_api
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = ppo_config or TextPPOConfig()
    methods = methods or METHODS
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    write_json(
        output_dir / "config.json",
        {
            "env_ids": env_ids,
            "methods": methods,
            "repetitions": repetitions,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "seed": seed,
            "min_mean_delta": min_mean_delta,
            "max_ci_low_regression": max_ci_low_regression,
            "ppo_config": ppo_config,
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

    metrics_rows: list[dict[str, Any]] = []
    records_by_method: dict[str, list[EpisodeRecord]] = {method: [] for method in methods}
    for repetition in range(repetitions):
        for method in methods:
            if method in POLICY_ABLATION_GATE_MODES:
                rows, test_records, _ = run_policy_ablation_once(
                    method=method,
                    env_ids=env_ids,
                    repetition=repetition,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    test_seeds=test_seeds,
                    seed=seed,
                    output_dir=output_dir,
                    min_mean_delta=min_mean_delta,
                    max_ci_low_regression=max_ci_low_regression,
                    ppo_config=ppo_config,
                )
            elif method == "textgrad_policy_iteration":
                rows, test_records, _ = run_policy_iteration_once(
                    env_ids=env_ids,
                    repetition=repetition,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    test_seeds=test_seeds,
                    seed=seed,
                    output_dir=output_dir,
                    min_mean_delta=min_mean_delta,
                    max_ci_low_regression=max_ci_low_regression,
                )
            elif method == "textgrad_ppo":
                rows, test_records, _ = run_textgrad_ppo_once(
                    env_ids=env_ids,
                    repetition=repetition,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    test_seeds=test_seeds,
                    seed=seed,
                    output_dir=output_dir,
                    min_mean_delta=min_mean_delta,
                    max_ci_low_regression=max_ci_low_regression,
                    ppo_config=ppo_config,
                )
            elif method == "textgrad_rl_plus":
                rows, test_records, _ = run_textgrad_plus_once(
                    env_ids=env_ids,
                    repetition=repetition,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    test_seeds=test_seeds,
                    seed=seed,
                    output_dir=output_dir,
                    min_mean_delta=min_mean_delta,
                    max_ci_low_regression=max_ci_low_regression,
                )
            else:
                rows, test_records, _ = run_baseline_once(
                    method=method,
                    env_ids=env_ids,
                    repetition=repetition,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    test_seeds=test_seeds,
                    seed=seed,
                    output_dir=output_dir,
                )
            metrics_rows.extend(rows)
            records_by_method[method].extend(test_records)

    ci_rows = build_ci_rows(records_by_method, seed)
    per_env_rows = build_per_env_rows(records_by_method)
    write_csv(output_dir / "metrics_by_run.csv", metrics_rows)
    write_csv(output_dir / "bootstrap_cis.csv", ci_rows)
    write_csv(output_dir / "per_env_metrics.csv", per_env_rows)
    write_summary(output_dir / "summary.md", env_ids, metrics_rows, ci_rows, methods)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextGrad policy-iteration benchmark.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--val-seeds", type=int, default=5)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--min-mean-delta", type=float, default=0.001)
    parser.add_argument("--max-ci-low-regression", type=float, default=0.0)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-target-kl", type=float, default=0.2)
    parser.add_argument("--ppo-score-scale", type=float, default=5.0)
    parser.add_argument("--ppo-min-surrogate-delta", type=float, default=0.001)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-dir", default="runs/textarena_policy_iteration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_ids = [env.strip() for env in args.envs.split(",") if env.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    output_dir = run_policy_iteration_suite(
        env_ids=env_ids,
        repetitions=args.repetitions,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        min_mean_delta=args.min_mean_delta,
        max_ci_low_regression=args.max_ci_low_regression,
        ppo_config=TextPPOConfig(
            clip_epsilon=args.ppo_clip_epsilon,
            target_kl=args.ppo_target_kl,
            score_scale=args.ppo_score_scale,
            min_surrogate_delta=args.ppo_min_surrogate_delta,
        ),
        methods=methods,
    )
    print(f"TextGrad policy-iteration artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
