"""Causal replay-gated TextGrad-RL+ benchmark for TextArena."""

from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
from dataclasses import dataclass
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
from textgrad_rl.benchmarks.textarena_paper_suite import (
    gradients_from_records,
    initial_modular_variables,
    scalar_prompt_search_update,
    slug_env,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


METHODS = [
    "fixed_prompt",
    "scalar_prompt_search",
    "modular_textgrad",
    "textgrad_rl_plus",
]

GENERIC_CANDIDATE_RULES = {
    "validity": (
        "Use only legal bracketed actions from the current observation, and avoid actions "
        "that just produced invalid-move feedback."
    ),
    "state_tracking": (
        "Track the latest visible game state after every observation before choosing the "
        "next action."
    ),
}


@dataclass
class CausalAssignment:
    env_id: str
    target_variable_name: str
    failure_count: int
    invalid_count: int
    average_reward: float
    causal_factor: str
    evidence: str


@dataclass
class CandidateEdit:
    name: str
    env_id: str
    target_variable_name: str
    rule: str
    variables: dict[str, TextVariable]
    gradients: list[TextualGradient]


def causal_assignments_from_records(records: list[EpisodeRecord]) -> list[CausalAssignment]:
    assignments: list[CausalAssignment] = []
    for env_id in sorted({record.env_id for record in records}):
        group = [record for record in records if record.env_id == env_id]
        failures = [record for record in group if record.reward < 1.0 or record.invalid_move]
        if not failures:
            continue
        invalid_count = sum(record.invalid_move for record in failures)
        target = f"{slug_env(canonical_env_id(env_id))}_strategy_prompt"
        factor = "action_format" if invalid_count > len(failures) / 2 else "environment_strategy"
        assignments.append(
            CausalAssignment(
                env_id=env_id,
                target_variable_name=target,
                failure_count=len(failures),
                invalid_count=invalid_count,
                average_reward=sum(record.reward for record in group) / len(group),
                causal_factor=factor,
                evidence=(
                    f"{len(failures)}/{len(group)} {env_id} replay trajectories had reward < 1.0 "
                    f"or invalid moves; invalid_count={invalid_count}."
                ),
            )
        )
    return assignments


def make_gradient(assignment: CausalAssignment, rule: str) -> TextualGradient:
    return TextualGradient(
        target_variable_name=assignment.target_variable_name,
        failure_mode=f"{assignment.env_id} {assignment.causal_factor} failure",
        evidence_from_trajectory=assignment.evidence,
        gradient_text=f"Repair the causal factor {assignment.causal_factor} for {assignment.env_id}.",
        suggested_edit=f"Add a rule: {rule}",
        confidence=0.9,
        forbidden_shortcuts=["hidden state", "environment mutation", "invalid action formats"],
    )


def generate_candidate_edits(
    assignment: CausalAssignment,
    variables: dict[str, TextVariable],
    optimizer: TextualGradientDescent,
) -> list[CandidateEdit]:
    """Generate multiple candidate text edits for one causal assignment."""

    candidates: list[CandidateEdit] = []
    env_rule = rule_for_env(assignment.env_id)
    gradient = make_gradient(assignment, env_rule)
    targeted = optimizer.step(
        variables,
        [gradient],
        constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
    )
    candidates.append(
        CandidateEdit(
            name="targeted_env_rule",
            env_id=assignment.env_id,
            target_variable_name=assignment.target_variable_name,
            rule=env_rule,
            variables=targeted,
            gradients=[gradient],
        )
    )

    for name, rule in GENERIC_CANDIDATE_RULES.items():
        generic_assignment = CausalAssignment(
            env_id=assignment.env_id,
            target_variable_name="general_strategy_prompt",
            failure_count=assignment.failure_count,
            invalid_count=assignment.invalid_count,
            average_reward=assignment.average_reward,
            causal_factor=assignment.causal_factor,
            evidence=assignment.evidence,
        )
        generic_gradient = make_gradient(generic_assignment, rule)
        generic = optimizer.step(
            variables,
            [generic_gradient],
            constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
        )
        candidates.append(
            CandidateEdit(
                name=f"generic_{name}",
                env_id=assignment.env_id,
                target_variable_name="general_strategy_prompt",
                rule=rule,
                variables=generic,
                gradients=[generic_gradient],
            )
        )
    return candidates


def record_score(record: EpisodeRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.5 * (1.0 if record.invalid_move else 0.0)
        - 0.005 * record.turns
    )


def bootstrap_delta_ci(
    old_records: list[EpisodeRecord],
    new_records: list[EpisodeRecord],
    seed: int,
    samples: int = 1000,
) -> tuple[float, float, float]:
    deltas = [record_score(new) - record_score(old) for old, new in zip(old_records, new_records)]
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(deltas) for _ in deltas]
        means.append(sum(draw) / len(draw))
    means.sort()
    mean = sum(deltas) / len(deltas)
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return mean, lo, hi


def gate_accepts(
    old_records: list[EpisodeRecord],
    new_records: list[EpisodeRecord],
    seed: int,
    min_mean_delta: float,
    max_ci_low_regression: float,
) -> tuple[bool, dict[str, Any]]:
    mean, lo, hi = bootstrap_delta_ci(old_records, new_records, seed)
    old_metrics = summarize(old_records)
    new_metrics = summarize(new_records)
    accepted = mean > min_mean_delta and lo >= -max_ci_low_regression
    return accepted, {
        "accepted": accepted,
        "mean_delta": mean,
        "ci_low": lo,
        "ci_high": hi,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "min_mean_delta": min_mean_delta,
        "max_ci_low_regression": max_ci_low_regression,
    }


def variables_changed(old: dict[str, TextVariable], new: dict[str, TextVariable]) -> bool:
    return any(old[name].value != new.get(name, old[name]).value for name in old)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def apply_rule_library(library: dict[str, Any]) -> dict[str, TextVariable]:
    variables = initial_modular_variables()
    for entry in library["accepted_rules"]:
        name = entry["target_variable_name"]
        if name not in variables:
            continue
        variable = variables[name]
        rule = entry["rule"].rstrip(".") + "."
        if rule.lower() in variable.value.lower():
            continue
        if "Learned rules:" not in variable.value:
            variable.value = variable.value.rstrip() + "\n\nLearned rules:"
        variable.value += f"\n- {rule}"
        variable.version += 1
        variable.gradient_history.append(
            f"retrieved_from_rule_library:{entry['env_id']}:{entry['candidate_name']}"
        )
    return variables


def run_baseline_once(
    method: str,
    env_ids: list[str],
    repetition: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[EpisodeRecord], dict[str, TextVariable]]:
    method_dir = output_dir / "methods" / method / f"rep_{repetition:03d}"
    games_dir = method_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    variables = initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    rows: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0
    candidate_count = 0

    if method != "fixed_prompt":
        train_records = run_records(
            env_ids,
            method,
            "train",
            train_seeds,
            seed + repetition * 10_000,
            variables,
            games_dir / "train.jsonl",
        )
        rows.append(row_for(method, repetition, "train", train_records))
        if method == "scalar_prompt_search":
            candidate = scalar_prompt_search_update(variables, train_records, repetition)
        else:
            gradients = gradients_from_records(train_records, variables)
            candidate = optimizer.step(
                variables,
                gradients,
                constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
            )
            write_json(method_dir / "gradients.json", gradients)
        candidate_count = 1
        old_val = run_records(
            env_ids,
            method,
            "val_old",
            val_seeds,
            seed + repetition * 10_000 + 1000,
            variables,
            games_dir / "val_old.jsonl",
        )
        new_val = run_records(
            env_ids,
            method,
            "val_candidate",
            val_seeds,
            seed + repetition * 10_000 + 1000,
            candidate,
            games_dir / "val_candidate.jsonl",
        )
        accepted = variables_changed(variables, candidate) and score_records(new_val) >= score_records(old_val)
        if accepted:
            variables = candidate
            accepted_count = 1
        else:
            rejected_count = 1
        append_jsonl(
            method_dir / ("accepted_updates.jsonl" if accepted else "rejected_updates.jsonl"),
            {
                "accepted": accepted,
                "old_metrics": summarize(old_val),
                "new_metrics": summarize(new_val),
            },
        )
        rows.append(row_for(method, repetition, "val_old", old_val, candidate_count=candidate_count))
        rows.append(
            row_for(
                method,
                repetition,
                "val_candidate",
                new_val,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                candidate_count=candidate_count,
            )
        )

    test_records = run_records(
        env_ids,
        method,
        "test",
        test_seeds,
        seed + repetition * 10_000 + 2000,
        variables,
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
    write_json(method_dir / "final_text_variables.json", variables)
    return rows, test_records, variables


def score_records(records: list[EpisodeRecord]) -> float:
    if not records:
        return 0.0
    return sum(record_score(record) for record in records) / len(records)


def run_textgrad_plus_once(
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
    method = "textgrad_rl_plus"
    method_dir = output_dir / "methods" / method / f"rep_{repetition:03d}"
    games_dir = method_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    variables = initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    base_seed = seed + repetition * 10_000
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
    rows.append(row_for(method, repetition, "train", train_records))
    assignments = causal_assignments_from_records(train_records)
    write_json(method_dir / "causal_assignments.json", assignments)

    for assignment_index, assignment in enumerate(assignments):
        best_candidate: CandidateEdit | None = None
        best_details: dict[str, Any] | None = None
        best_delta = float("-inf")
        candidates = generate_candidate_edits(assignment, variables, optimizer)
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
            accepted, details = gate_accepts(
                old_replay + old_val,
                new_replay + new_val,
                seed=base_seed + assignment_index * 101 + candidate_index,
                min_mean_delta=min_mean_delta,
                max_ci_low_regression=max_ci_low_regression,
            )
            changed = variables_changed(variables, candidate.variables)
            details.update(
                {
                    "candidate_name": candidate.name,
                    "env_id": candidate.env_id,
                    "target_variable_name": candidate.target_variable_name,
                    "rule": candidate.rule,
                    "changed": changed,
                }
            )
            append_jsonl(method_dir / "candidate_evaluations.jsonl", details)
            if accepted and changed and details["mean_delta"] > best_delta:
                best_candidate = candidate
                best_details = details
                best_delta = details["mean_delta"]

        if best_candidate is None or best_details is None:
            rejected_count += len(candidates)
            library["rejected_rules"].append(
                {
                    "env_id": assignment.env_id,
                    "target_variable_name": assignment.target_variable_name,
                    "reason": "no candidate passed replay-bootstrap gate",
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
            }
        )
        append_jsonl(method_dir / "accepted_updates.jsonl", best_details)

    library_variables = apply_rule_library(library)
    write_json(method_dir / "learned_rule_library.json", library)
    write_json(method_dir / "final_text_variables.json", library_variables)
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


def bootstrap_ci(values: list[float], seed: int, samples: int = 1000) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    mean = sum(values) / len(values)
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return mean, lo, hi


def build_per_env_rows(records_by_method: dict[str, list[EpisodeRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, records in sorted(records_by_method.items()):
        for env_id in sorted({record.env_id for record in records}):
            group = [record for record in records if record.env_id == env_id]
            rows.append({"method": method, "env_id": env_id, **summarize(group)})
    return rows


def write_summary(
    path: Path,
    env_ids: list[str],
    metrics_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
) -> None:
    test_rows = [row for row in metrics_rows if row["split"] == "test"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        grouped.setdefault(row["method"], []).append(row)
    lines = [
        "# TextArena TextGrad-RL+ Suite",
        "",
        f"- Environments: {len(env_ids)}",
        f"- Methods: {', '.join(METHODS)}",
        "",
        "## Implemented Improvements",
        "",
        "- Multi-candidate textual optimization: targeted and generic candidates per causal assignment.",
        "- Causal credit assignment: failed trajectories target environment-specific text variables.",
        "- Replay validation: candidates are scored on train replay plus validation seeds.",
        "- Uncertainty-aware gate: bootstrap delta CI with a minimum mean-gain threshold.",
        "- Learned rule library: accepted rules are stored and retrieved into final test prompts.",
        "",
        "## Test Means Across Repetitions",
        "",
        "| method | reward | success | invalid | turns | accepted | candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
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


def run_textgrad_plus_suite(
    env_ids: list[str],
    repetitions: int,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    min_mean_delta: float,
    max_ci_low_regression: float,
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
            "env_ids": env_ids,
            "methods": METHODS,
            "repetitions": repetitions,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "seed": seed,
            "min_mean_delta": min_mean_delta,
            "max_ci_low_regression": max_ci_low_regression,
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
    records_by_method: dict[str, list[EpisodeRecord]] = {method: [] for method in METHODS}
    for repetition in range(repetitions):
        for method in METHODS:
            if method == "textgrad_rl_plus":
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
    write_summary(output_dir / "summary.md", env_ids, metrics_rows, ci_rows)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextArena TextGrad-RL+ benchmark.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--val-seeds", type=int, default=5)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--min-mean-delta", type=float, default=0.001)
    parser.add_argument("--max-ci-low-regression", type=float, default=0.0)
    parser.add_argument("--output-dir", default="runs/textarena_textgrad_plus")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_ids = [env.strip() for env in args.envs.split(",") if env.strip()]
    output_dir = run_textgrad_plus_suite(
        env_ids=env_ids,
        repetitions=args.repetitions,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        min_mean_delta=args.min_mean_delta,
        max_ci_low_regression=args.max_ci_low_regression,
    )
    print(f"TextArena TextGrad-RL+ artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
