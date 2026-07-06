"""Paper-oriented TextArena suite with baselines, CIs, and examples."""

from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
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
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


METHODS = [
    "fixed_prompt",
    "scalar_prompt_search",
    "monolithic_textgrad",
    "modular_textgrad",
    "no_acceptance_gate",
]

SCALAR_RULES = [
    "Choose only syntactically valid actions shown or implied by the latest observation.",
    "Avoid repeating an action that just produced an invalid-move or no-progress outcome.",
    "Prefer simple lookahead before committing to a board-game move.",
    "For single-player puzzles, update an internal state estimate after every observation.",
]


def slug_env(env_id: str) -> str:
    return env_id.replace("-v0", "").replace("-", "_").lower()


def initial_modular_variables() -> dict[str, TextVariable]:
    variables = {
        "general_strategy_prompt": TextVariable(
            name="general_strategy_prompt",
            value=(
                "Play valid TextArena actions. Read the latest observation, choose legal "
                "moves, and avoid changing the environment rules."
            ),
            role_description="General TextArena action-format and safety strategy.",
            max_chars=2400,
        )
    }
    for env_id in DEFAULT_ENVS:
        name = f"{slug_env(env_id)}_strategy_prompt"
        variables[name] = TextVariable(
            name=name,
            value=f"For {env_id}, start with a simple legal action policy.",
            role_description=f"Environment-specific strategy text for {env_id}.",
            max_chars=1600,
        )
    return variables


def initial_monolithic_variables() -> dict[str, TextVariable]:
    value = "Play valid TextArena actions across all environments. Use simple legal moves.\n"
    for env_id in DEFAULT_ENVS:
        value += f"\n[{env_id}] Start with a simple legal action policy."
    return {
        "monolithic_textarena_prompt": TextVariable(
            name="monolithic_textarena_prompt",
            value=value,
            role_description="Single prompt containing every TextArena environment strategy.",
            max_chars=8000,
        )
    }


def gradients_from_records(
    records: list[EpisodeRecord],
    text_variables: dict[str, TextVariable],
) -> list[TextualGradient]:
    failed_envs = sorted({record.env_id for record in records if record.reward < 1.0 or record.invalid_move})
    gradients: list[TextualGradient] = []
    monolithic = "monolithic_textarena_prompt" in text_variables
    for env_id in failed_envs:
        target = "monolithic_textarena_prompt" if monolithic else f"{slug_env(canonical_env_id(env_id))}_strategy_prompt"
        gradients.append(
            TextualGradient(
                target_variable_name=target,
                failure_mode=f"{env_id} trajectory reward below target",
                evidence_from_trajectory=(
                    f"Training trajectories for {env_id} included sub-solved reward or invalid moves."
                ),
                gradient_text=f"The agent needs an environment-specific repair rule for {env_id}.",
                suggested_edit=f"Add a rule: {rule_for_env(env_id)}",
                confidence=0.85,
                forbidden_shortcuts=["hidden state", "environment mutation", "invalid action formats"],
            )
        )
    return gradients


def scalar_prompt_search_update(
    variables: dict[str, TextVariable],
    train_records: list[EpisodeRecord],
    repetition: int,
) -> dict[str, TextVariable]:
    updated = deepcopy(variables)
    target_name = "monolithic_textarena_prompt" if "monolithic_textarena_prompt" in updated else "general_strategy_prompt"
    variable = updated[target_name]
    failed = sum(record.reward < 1.0 or record.invalid_move for record in train_records)
    rule = SCALAR_RULES[(failed + repetition) % len(SCALAR_RULES)]
    if rule.lower() not in variable.value.lower():
        if "Learned rules:" not in variable.value:
            variable.value = variable.value.rstrip() + "\n\nLearned rules:"
        variable.value += f"\n- {rule}"
        variable.version += 1
        variable.gradient_history.append(f"scalar_prompt_search: failed_episodes={failed}")
    return updated


def variables_changed(old: dict[str, TextVariable], new: dict[str, TextVariable]) -> bool:
    return any(
        old[name].value != new.get(name, old[name]).value
        or old[name].version != new.get(name, old[name]).version
        for name in old
    )


def score(records: list[EpisodeRecord]) -> float:
    metrics = summarize(records)
    return (
        3.0 * metrics["average_reward"]
        + 2.0 * metrics["success_rate"]
        - 2.0 * metrics["invalid_move_rate"]
        - 0.01 * metrics["average_turns"]
    )


def run_method_once(
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
    variables = initial_monolithic_variables() if method == "monolithic_textgrad" else initial_modular_variables()
    optimizer = TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=20)
    rows: list[dict[str, Any]] = []
    accepted = False
    gradient_count = 0

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
        rows.append(row_for(method, repetition, "train", train_records, accepted=False, gradient_count=0))
        if method == "scalar_prompt_search":
            candidate = scalar_prompt_search_update(variables, train_records, repetition)
            gradients: list[TextualGradient] = []
        else:
            gradients = gradients_from_records(train_records, variables)
            candidate = optimizer.step(
                variables,
                gradients,
                constraints=["Do not use hidden state", "Do not mutate TextArena environments"],
            )
        gradient_count = len(gradients)
        write_json(method_dir / "gradients.json", gradients)

        if method == "no_acceptance_gate":
            accepted = variables_changed(variables, candidate)
            variables = candidate
            append_jsonl(method_dir / "accepted_updates.jsonl", {"accepted": accepted, "reason": "no gate"})
        else:
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
            changed = variables_changed(variables, candidate)
            accepted = changed and score(new_val) >= score(old_val)
            if accepted:
                variables = candidate
            append_jsonl(
                method_dir / ("accepted_updates.jsonl" if accepted else "rejected_updates.jsonl"),
                {
                    "accepted": accepted,
                    "changed": changed,
                    "old_score": score(old_val),
                    "new_score": score(new_val),
                    "old_metrics": summarize(old_val),
                    "new_metrics": summarize(new_val),
                },
            )
            rows.append(row_for(method, repetition, "val_old", old_val, accepted=False, gradient_count=gradient_count))
            rows.append(row_for(method, repetition, "val_candidate", new_val, accepted=accepted, gradient_count=gradient_count))

    test_records = run_records(
        env_ids,
        method,
        "test",
        test_seeds,
        seed + repetition * 10_000 + 2000,
        variables,
        games_dir / "test.jsonl",
    )
    rows.append(row_for(method, repetition, "test", test_records, accepted=accepted, gradient_count=gradient_count))
    write_json(method_dir / "final_text_variables.json", variables)
    return rows, test_records, variables


def row_for(
    method: str,
    repetition: int,
    split: str,
    records: list[EpisodeRecord],
    accepted: bool,
    gradient_count: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "repetition": repetition,
        "split": split,
        "accepted": accepted,
        "gradient_count": gradient_count,
        **summarize(records),
    }


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
) -> None:
    test_rows = [row for row in metrics_rows if row["split"] == "test"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        grouped.setdefault(row["method"], []).append(row)
    lines = [
        "# TextArena Paper Suite",
        "",
        f"- Environments: {len(env_ids)}",
        f"- Methods: {', '.join(METHODS)}",
        "",
        "## Test Means Across Repetitions",
        "",
        "| method | reward | success | invalid | turns |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        rows = grouped.get(method, [])
        if not rows:
            continue
        lines.append(
            "| {method} | {reward:.3f} | {success:.3f} | {invalid:.3f} | {turns:.3f} |".format(
                method=method,
                reward=sum(row["average_reward"] for row in rows) / len(rows),
                success=sum(row["success_rate"] for row in rows) / len(rows),
                invalid=sum(row["invalid_move_rate"] for row in rows) / len(rows),
                turns=sum(row["average_turns"] for row in rows) / len(rows),
            )
        )
    lines.extend(["", "## Bootstrap 95% CIs", "", "| method | metric | mean | ci_low | ci_high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in ci_rows:
        if row["metric"] in {"reward", "success", "invalid_move"}:
            lines.append(
                "| {method} | {metric} | {mean:.3f} | {ci_low:.3f} | {ci_high:.3f} | {n} |".format(**row)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualitative_examples(path: Path, output_dir: Path) -> None:
    lines = ["# Qualitative Examples", ""]
    for gradient_file in sorted(output_dir.glob("methods/*/rep_*/gradients.json")):
        import json

        gradients = json.loads(gradient_file.read_text(encoding="utf-8"))
        if not gradients:
            continue
        lines.append(f"## {gradient_file.parent.parent.name} {gradient_file.parent.name}")
        lines.append("")
        gradient = gradients[0]
        lines.append(f"- Failure mode: {gradient.get('failure_mode')}")
        lines.append(f"- Evidence: {gradient.get('evidence_from_trajectory')}")
        lines.append(f"- Suggested edit: {gradient.get('suggested_edit')}")
        lines.append("")
        if len(lines) > 30:
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_paper_suite(
    env_ids: list[str],
    repetitions: int,
    train_seeds: int,
    val_seeds: int,
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
            rows, test_records, _ = run_method_once(
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
    write_csv(output_dir / "metrics_by_run.csv", metrics_rows)
    write_csv(output_dir / "bootstrap_cis.csv", ci_rows)
    write_summary(output_dir / "summary.md", env_ids, metrics_rows, ci_rows)
    write_qualitative_examples(output_dir / "qualitative_examples.md", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-oriented TextArena ablations.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--val-seeds", type=int, default=5)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--output-dir", default="runs/textarena_paper_suite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_ids = [env.strip() for env in args.envs.split(",") if env.strip()]
    output_dir = run_paper_suite(
        env_ids=env_ids,
        repetitions=args.repetitions,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(f"TextArena paper suite artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
