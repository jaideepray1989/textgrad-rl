"""Run compact controls requested during review of the RulePI paper."""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_broad_suite import (
    BROAD_TEXTARENA_ENVS,
    run_broad_episode,
    run_broad_records,
)
from textgrad_rl.benchmarks.textarena_multienv_compare import DEFAULT_ENVS, ENV_RULES
from textgrad_rl.benchmarks.textarena_paper_suite import initial_modular_variables, slug_env
from textgrad_rl.benchmarks.textarena_textgrad_plus import GENERIC_CANDIDATE_RULES, apply_rule_library
from textgrad_rl.benchmarks.textworld_24_suite import (
    RULE_TEXTS,
    default_specs,
    gradients_from_records,
    initial_textworld_variables,
    learned_rule_ids,
    run_episode,
    run_records,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "rulepi_reviewer_controls"
TEXTWORLD_DIR = ROOT / "runs" / "rulepi_textworld_10seed"
TEXTARENA_DIR = ROOT / "runs" / "rulepi_textarena_supported_10seed"
RANDOM_POLICY_SEEDS = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def mean(rows: list[Any], field: str) -> float:
    def value(row: Any) -> float:
        raw = getattr(row, field) if hasattr(row, field) else row.get(field, 0)
        return float(raw)

    return statistics.mean(value(row) for row in rows)


def textworld_all_on_variables() -> dict[str, TextVariable]:
    variables = initial_textworld_variables()
    gradients = [
        TextualGradient(
            target_variable_name="textworld_policy",
            failure_mode=f"always_on:{rule_id}",
            evidence_from_trajectory="Always-on control; no trajectory used.",
            gradient_text="Enable the complete predefined library.",
            suggested_edit=f"Add a rule: {rule}",
            confidence=1.0,
        )
        for rule_id, rule in RULE_TEXTS.items()
    ]
    return TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=4).step(
        variables, gradients, constraints=[]
    )


def run_textworld_controls() -> list[dict[str, Any]]:
    output = OUTPUT_DIR / "textworld"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(TEXTWORLD_DIR / "run_manifest.csv")
    all_on_records = []
    memory_records = []
    memory_rule_counts = []

    for manifest_row in manifest:
        repetition = int(manifest_row["repetition"])
        run_dir = ROOT / manifest_row["run_dir"]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        seed = int(config["test_specs"][0]["seed"])
        specs = default_specs(seed)
        game_paths = {spec.spec_id: run_dir / "games" / f"{spec.spec_id}.z8" for spec in specs}

        records = run_records(
            specs=specs,
            game_paths=game_paths,
            method="always_on_rules",
            split="test",
            variables=textworld_all_on_variables(),
            max_steps=80,
            output_jsonl=output / f"repetition_{repetition:02d}_always_on.jsonl",
        )
        all_on_records.extend((repetition, record) for record in records)

        variables = initial_textworld_variables()
        memory_path = output / f"repetition_{repetition:02d}_retry_persistent.jsonl"
        if memory_path.exists():
            memory_path.unlink()
        for spec in specs:
            first = run_episode(
                spec=spec,
                game_path=game_paths[spec.spec_id],
                method="retry_with_persistence",
                split="test_attempt_1",
                variables=variables,
                max_steps=80,
            )
            if first.success:
                final = replace(first, split="test")
            else:
                gradients = gradients_from_records([first])
                variables = TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=4).step(
                    variables,
                    gradients,
                    constraints=[
                        "must not use oracle walkthrough",
                        "must not use policy_commands",
                        "must not inspect hidden labels",
                    ],
                )
                second = run_episode(
                    spec=spec,
                    game_path=game_paths[spec.spec_id],
                    method="retry_with_persistence",
                    split="test_attempt_2",
                    variables=variables,
                    max_steps=80,
                )
                final = replace(
                    second,
                    split="test",
                    attempts=2,
                    total_turns=first.turns + second.turns,
                )
            memory_records.append((repetition, final))
            append_jsonl(memory_path, final)
        memory_rule_counts.append(len(learned_rule_ids(variables)))

    existing = read_csv(TEXTWORLD_DIR / "all_test_records.csv")
    rows = []
    for method in ["fixed_prompt", "retry_with_diagnostics", "textgrad_policy_iteration"]:
        group = [row for row in existing if row["method"] == method]
        rows.append(
            {
                "benchmark": "TextWorld-24",
                "method": method,
                "episodes": len(group),
                "success": statistics.mean(as_bool(row["success"]) for row in group),
                "attempts": statistics.mean(float(row["attempts"]) for row in group),
                "test_actions": statistics.mean(float(row["total_turns"]) for row in group),
                "rules_enabled": "0" if method == "fixed_prompt" else "task-local" if method == "retry_with_diagnostics" else "3",
            }
        )
    for method, pairs, rule_count in [
        ("always_on_rules", all_on_records, "3"),
        ("retry_with_persistence", memory_records, f"{statistics.mean(memory_rule_counts):.1f}"),
    ]:
        group = [record for _repetition, record in pairs]
        rows.append(
            {
                "benchmark": "TextWorld-24",
                "method": method,
                "episodes": len(group),
                "success": mean(group, "success"),
                "attempts": mean(group, "attempts"),
                "test_actions": mean(group, "total_turns"),
                "rules_enabled": rule_count,
            }
        )
    return rows


def textarena_library_entries() -> list[dict[str, str]]:
    entries = [
        {
            "env_id": env_id,
            "target_variable_name": f"{slug_env(env_id)}_strategy_prompt",
            "candidate_name": "environment_rule",
            "rule": rule,
        }
        for env_id, rule in ENV_RULES.items()
    ]
    entries.extend(
        {
            "env_id": "generic",
            "target_variable_name": "general_strategy_prompt",
            "candidate_name": name,
            "rule": rule,
        }
        for name, rule in GENERIC_CANDIDATE_RULES.items()
    )
    return entries


def variables_from_entries(entries: list[dict[str, str]]) -> dict[str, TextVariable]:
    return apply_rule_library({"accepted_rules": entries})


def add_textarena_rule(variables: dict[str, TextVariable], env_id: str) -> dict[str, TextVariable]:
    rule = ENV_RULES[env_id]
    gradient = TextualGradient(
        target_variable_name=f"{slug_env(env_id)}_strategy_prompt",
        failure_mode=f"test_memory:{env_id}",
        evidence_from_trajectory="The current deployment episode failed.",
        gradient_text="Persist the environment rule for later deployment tasks.",
        suggested_edit=f"Add a rule: {rule}",
        confidence=0.85,
    )
    return TextualGradientDescent(max_prompt_chars=8000, max_rules_per_step=1).step(
        variables, [gradient], constraints=[]
    )


def run_textarena_controls() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = OUTPUT_DIR / "textarena"
    output.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in BROAD_TEXTARENA_ENVS if spec.env_id in set(DEFAULT_ENVS)]
    entries = textarena_library_entries()

    all_on = run_broad_records(
        specs=specs,
        method="always_on_rules",
        split="test",
        seeds_per_env=10,
        seed=117001,
        variables=variables_from_entries(entries),
        turn_budget=80,
        output_jsonl=output / "always_on.jsonl",
    )

    variables = initial_modular_variables()
    memory = []
    memory_path = output / "retry_persistent.jsonl"
    if memory_path.exists():
        memory_path.unlink()
    for spec in specs:
        target_sides = [None] if spec.num_players == 1 else list(range(spec.num_players))
        for target_side in target_sides:
            for index in range(10):
                seed = 117001 + index
                first = run_broad_episode(
                    spec=spec,
                    method="retry_with_persistence",
                    split="test_attempt_1",
                    seed=seed,
                    target_side=target_side,
                    variables=variables,
                    turn_budget=80,
                )
                if first.success:
                    final = replace(first, split="test")
                else:
                    variables = add_textarena_rule(variables, spec.env_id)
                    second = run_broad_episode(
                        spec=spec,
                        method="retry_with_persistence",
                        split="test_attempt_2",
                        seed=seed,
                        target_side=target_side,
                        variables=variables,
                        turn_budget=80,
                    )
                    final = replace(
                        second,
                        split="test",
                        attempts=2,
                        total_turns=first.turns + second.turns,
                    )
                memory.append(final)
                append_jsonl(memory_path, final)

    random_rows = []
    random_policy_summaries = []
    active_entries = [entry for entry in entries if entry["env_id"] != "generic"]
    for policy_seed in range(RANDOM_POLICY_SEEDS):
        selected = random.Random(88000 + policy_seed).sample(active_entries, 6)
        records = run_broad_records(
            specs=specs,
            method="random_library_subset",
            split="test",
            seeds_per_env=10,
            seed=117001,
            variables=variables_from_entries(selected),
            turn_budget=80,
            output_jsonl=output / f"random_{policy_seed:02d}.jsonl",
        )
        summary = {
            "policy_seed": policy_seed,
            "selected_rules": ";".join(f"{entry['env_id']}:{entry['candidate_name']}" for entry in selected),
            "success": mean(records, "success"),
            "attempts": mean(records, "attempts"),
            "test_actions": mean(records, "total_turns"),
        }
        random_policy_summaries.append(summary)
        random_rows.extend(records)

    existing = {row["method"]: row for row in read_csv(TEXTARENA_DIR / "summary.csv")}
    rows = []
    for method in ["fixed_prompt", "retry_with_diagnostics", "textgrad_policy_iteration"]:
        row = existing[method]
        rows.append(
            {
                "benchmark": "TextArena-10",
                "method": method,
                "episodes": int(row["episodes"]),
                "success": float(row["success_rate"]),
                "attempts": float(row["average_attempts"]),
                "test_actions": float(row["average_test_actions"]),
                "rules_enabled": "0" if method == "fixed_prompt" else "task-local" if method == "retry_with_diagnostics" else "6",
            }
        )
    for method, group, rule_count in [
        ("always_on_rules", all_on, "12"),
        ("retry_with_persistence", memory, "test-discovered"),
    ]:
        rows.append(
            {
                "benchmark": "TextArena-10",
                "method": method,
                "episodes": len(group),
                "success": mean(group, "success"),
                "attempts": mean(group, "attempts"),
                "test_actions": mean(group, "total_turns"),
                "rules_enabled": rule_count,
            }
        )
    rows.append(
        {
            "benchmark": "TextArena-10",
            "method": "random_library_subset",
            "episodes": len(random_rows),
            "success": statistics.mean(row["success"] for row in random_policy_summaries),
            "attempts": 1.0,
            "test_actions": statistics.mean(row["test_actions"] for row in random_policy_summaries),
            "rules_enabled": "6 random env rules (30 policies)",
        }
    )
    return rows, random_policy_summaries


def textworld_record_score(record: dict[str, Any], metric: str) -> float:
    success = float(as_bool(record["success"]))
    if metric == "success":
        return success
    if metric == "reward":
        return float(record["reward"])
    if metric == "reward_success":
        return float(record["reward"]) + 0.5 * success
    return (
        float(record["reward"])
        + 0.5 * success
        - 0.4 * float(as_bool(record["invalid_action"]))
        - 0.1 * float(as_bool(record["repeated_actions"]))
        - 0.004 * float(record["turns"])
    )


def gate_sensitivity() -> list[dict[str, Any]]:
    manifest = read_csv(TEXTWORLD_DIR / "run_manifest.csv")
    test_records = read_csv(TEXTWORLD_DIR / "all_test_records.csv")
    margins: dict[str, dict[int, float]] = {
        metric: {} for metric in ["success", "reward", "reward_success", "default"]
    }
    protected: dict[int, bool] = {}
    for manifest_row in manifest:
        repetition = int(manifest_row["repetition"])
        run_dir = ROOT / manifest_row["run_dir"] / "textgrad_policy_iteration"
        old = read_jsonl(run_dir / "val_base.jsonl")
        new = read_jsonl(run_dir / "val_candidate.jsonl")
        for metric in margins:
            margins[metric][repetition] = statistics.mean(
                textworld_record_score(new_row, metric) - textworld_record_score(old_row, metric)
                for old_row, new_row in zip(old, new)
            )
        families = sorted({row["family"] for row in old})
        family_deltas = [
            statistics.mean(
                textworld_record_score(new_row, "default") - textworld_record_score(old_row, "default")
                for old_row, new_row in zip(old, new)
                if old_row["family"] == family
            )
            for family in families
        ]
        protected[repetition] = min(family_deltas) >= 0.0

    settings = [
        ("default", 0.0, None),
        ("default", 0.001, None),
        ("default", 0.05, None),
        ("default", 0.10, None),
        ("default", 0.20, None),
        ("success", 0.001, None),
        ("reward", 0.001, None),
        ("reward_success", 0.001, None),
        ("default", 0.001, protected),
    ]
    rows = []
    for metric, delta, override in settings:
        accepted = override or {
            repetition: margin >= delta for repetition, margin in margins[metric].items()
        }
        deployed = []
        for record in test_records:
            repetition = int(record["repetition"])
            method = "textgrad_policy_iteration" if accepted[repetition] else "fixed_prompt"
            if record["method"] == method:
                deployed.append(record)
        rows.append(
            {
                "metric": metric,
                "delta": delta,
                "protected_family_nonregression": override is not None,
                "accepted_repetitions": sum(accepted.values()),
                "test_success": statistics.mean(as_bool(row["success"]) for row in deployed),
                "test_actions": statistics.mean(float(row["total_turns"]) for row in deployed),
            }
        )
    return rows


def idempotence_check() -> dict[str, Any]:
    run_dir = ROOT / read_csv(TEXTWORLD_DIR / "run_manifest.csv")[0]["run_dir"]
    train = read_jsonl(run_dir / "textgrad_policy_iteration" / "train_base.jsonl")
    gradients = gradients_from_records([type("Record", (), row)() for row in train])
    optimizer = TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=4)
    first = optimizer.step(initial_textworld_variables(), gradients, constraints=[])
    second = optimizer.step(first, gradients, constraints=[])
    return {
        "first_rule_count": len(learned_rule_ids(first)),
        "second_rule_count": len(learned_rule_ids(second)),
        "policy_text_changed": first["textworld_policy"].value != second["textworld_policy"].value,
        "first_version": first["textworld_policy"].version,
        "second_version": second["textworld_policy"].version,
    }


def write_summary(
    controls: list[dict[str, Any]],
    random_policies: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    idempotence: dict[str, Any],
) -> None:
    lines = [
        "# RulePI Reviewer Controls",
        "",
        "## Policy Controls",
        "",
        "| Benchmark | Method | Episodes | Success | Attempts | Test actions | Rules |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in controls:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['episodes']} | "
            f"{100 * row['success']:.1f}% | {row['attempts']:.2f} | "
            f"{row['test_actions']:.2f} | {row['rules_enabled']} |"
        )
    random_success = sorted(row["success"] for row in random_policies)
    lines.extend(
        [
            "",
            f"TextArena random-subset control uses {len(random_policies)} independently sampled six-rule policies. "
            f"Across policies, success ranges from {100 * random_success[0]:.1f}% to "
            f"{100 * random_success[-1]:.1f}% (SD {100 * statistics.stdev(random_success):.1f} points).",
            "",
            "## TextWorld Gate Sensitivity (Post-hoc)",
            "",
            "| Metric | delta | Family protection | Accepted / 10 | Test success | Test actions |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in sensitivity:
        lines.append(
            f"| {row['metric']} | {row['delta']:.3f} | "
            f"{'yes' if row['protected_family_nonregression'] else 'no'} | "
            f"{row['accepted_repetitions']} | {100 * row['test_success']:.1f}% | "
            f"{row['test_actions']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-Library Second Pass",
            "",
            f"A second TextWorld pass retains {idempotence['second_rule_count']} rules after the first pass retained "
            f"{idempotence['first_rule_count']}; policy text changed: {idempotence['policy_text_changed']}. "
            f"Variable version remains {idempotence['first_version']} -> {idempotence['second_version']}.",
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    textworld = run_textworld_controls()
    textarena, random_policies = run_textarena_controls()
    sensitivity = gate_sensitivity()
    idempotence = idempotence_check()
    controls = textworld + textarena
    write_csv(OUTPUT_DIR / "policy_controls.csv", controls)
    write_csv(OUTPUT_DIR / "random_textarena_policies.csv", random_policies)
    write_csv(OUTPUT_DIR / "gate_sensitivity.csv", sensitivity)
    write_json(OUTPUT_DIR / "idempotence.json", idempotence)
    write_summary(controls, random_policies, sensitivity, idempotence)
    print(f"Reviewer controls saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
