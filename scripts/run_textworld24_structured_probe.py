"""Full TextWorld-24 structured controller probe.

This extends the one-game structured-rule probe across all 24 local TextWorld
games and compares base controller rules against TextGrad-style structured
controller edits, with optional qwen ranking over controller candidates.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_textworld_simple1_structured_probe import ProbeRecord, run_structured_variant
from textgrad_rl.benchmarks.textarena_slm_compare import OpenAICompatibleChatModel
from textgrad_rl.benchmarks.textworld_24_suite import TextWorldSpec, default_specs, ensure_games
from textgrad_rl.utils.json_utils import write_json


VARIANT_CONFIGS = {
    "controller_direct_base": {"use_qwen_ranker": False, "textgrad_rule": False},
    "controller_direct_textgrad_rule": {"use_qwen_ranker": False, "textgrad_rule": True},
    "qwen_ranker_base": {"use_qwen_ranker": True, "textgrad_rule": False},
    "qwen_ranker_textgrad_rule": {"use_qwen_ranker": True, "textgrad_rule": True},
}


@dataclass
class SuiteRecord:
    benchmark: str
    spec_id: str
    family: str
    category: str
    seed: int
    variant: str
    model: str
    temperature: float
    success: bool
    reward: float
    final_score: float
    max_score: float
    turns: int
    invalid: bool
    repeated: bool
    timeout_or_parse_failures: int
    qwen_calls: int
    fallback_calls: int
    runtime_seconds: float
    failure_reason: str
    actions: list[str]
    raw_outputs: list[str]


def suite_record(
    *,
    spec: TextWorldSpec,
    record: ProbeRecord,
    model: str,
    temperature: float,
) -> SuiteRecord:
    return SuiteRecord(
        benchmark="textworld_24_structured_probe",
        spec_id=spec.spec_id,
        family=spec.family,
        category=spec.category,
        seed=spec.seed,
        variant=record.variant,
        model=model,
        temperature=temperature,
        success=record.success,
        reward=record.reward,
        final_score=record.final_score,
        max_score=record.max_score,
        turns=record.turns,
        invalid=record.invalid,
        repeated=record.repeated,
        timeout_or_parse_failures=record.timeout_or_parse_failures,
        qwen_calls=record.qwen_calls,
        fallback_calls=record.fallback_calls,
        runtime_seconds=record.runtime_seconds,
        failure_reason=record.failure_reason,
        actions=record.actions,
        raw_outputs=record.raw_outputs,
    )


def summarize_records(records: list[SuiteRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SuiteRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.family, record.variant)].append(record)
    rows: list[dict[str, Any]] = []
    for (family, variant), items in sorted(grouped.items()):
        rows.append(aggregate_row(family, variant, items))
    for variant in sorted({record.variant for record in records}):
        rows.append(aggregate_row("ALL", variant, [record for record in records if record.variant == variant]))
    return rows


def aggregate_row(family: str, variant: str, items: list[SuiteRecord]) -> dict[str, Any]:
    successes = sum(record.success for record in items)
    n = len(items)
    return {
        "family": family,
        "variant": variant,
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "successes": successes,
        "avg_reward": sum(record.reward for record in items) / n if n else 0.0,
        "avg_turns": sum(record.turns for record in items) / n if n else 0.0,
        "avg_qwen_calls": sum(record.qwen_calls for record in items) / n if n else 0.0,
        "fallback_calls": sum(record.fallback_calls for record in items),
        "invalids": sum(record.invalid for record in items),
    }


def per_problem_rows(records: list[SuiteRecord]) -> list[dict[str, Any]]:
    variants = list(VARIANT_CONFIGS)
    by_problem: dict[str, dict[str, SuiteRecord]] = defaultdict(dict)
    specs: dict[str, SuiteRecord] = {}
    for record in records:
        by_problem[record.spec_id][record.variant] = record
        specs.setdefault(record.spec_id, record)
    rows: list[dict[str, Any]] = []
    for spec_id in sorted(by_problem):
        first = specs[spec_id]
        row: dict[str, Any] = {
            "spec_id": spec_id,
            "family": first.family,
            "category": first.category,
        }
        for variant in variants:
            record = by_problem[spec_id].get(variant)
            row[f"{variant}_success"] = int(record.success) if record else ""
            row[f"{variant}_reward"] = f"{record.reward:.3f}" if record else ""
            row[f"{variant}_turns"] = record.turns if record else ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def variant_cell(record: SuiteRecord | None) -> str:
    if record is None:
        return "-"
    return f"{int(record.success)} / {record.reward:.2f} / {record.turns}"


def write_markdown(path: Path, records: list[SuiteRecord], aggregate: list[dict[str, Any]]) -> None:
    by_problem: dict[str, dict[str, SuiteRecord]] = defaultdict(dict)
    first_by_problem: dict[str, SuiteRecord] = {}
    for record in records:
        by_problem[record.spec_id][record.variant] = record
        first_by_problem.setdefault(record.spec_id, record)

    lines = [
        "# TextWorld-24 Structured TextGrad Probe",
        "",
        "Each cell in the per-problem table is `success / reward / turns`.",
        "",
        "## Overall and Family Aggregates",
        "",
        "| family | variant | n | success rate | avg reward | avg turns | avg qwen calls | fallbacks | invalids |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['family']} | {row['variant']} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['avg_reward']:.3f} | {row['avg_turns']:.2f} | {row['avg_qwen_calls']:.2f} | "
            f"{row['fallback_calls']} | {row['invalids']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Problem Results",
            "",
            "| problem | family | direct base | direct TextGrad | qwen base | qwen TextGrad |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for spec_id in sorted(by_problem):
        first = first_by_problem[spec_id]
        variants = by_problem[spec_id]
        lines.append(
            f"| {spec_id} | {first.family} | "
            f"{variant_cell(variants.get('controller_direct_base'))} | "
            f"{variant_cell(variants.get('controller_direct_textgrad_rule'))} | "
            f"{variant_cell(variants.get('qwen_ranker_base'))} | "
            f"{variant_cell(variants.get('qwen_ranker_textgrad_rule'))} |"
        )

    lines.extend(["", "## Policy-Pair Deltas", ""])
    for base, improved, label in [
        ("controller_direct_base", "controller_direct_textgrad_rule", "direct controller"),
        ("qwen_ranker_base", "qwen_ranker_textgrad_rule", "qwen-ranked controller"),
    ]:
        base_items = {record.spec_id: record for record in records if record.variant == base}
        improved_items = {record.spec_id: record for record in records if record.variant == improved}
        paired = [(base_items[key], improved_items[key]) for key in sorted(base_items.keys() & improved_items.keys())]
        if not paired:
            continue
        success_delta = sum(new.success - old.success for old, new in paired) / len(paired)
        reward_delta = sum(new.reward - old.reward for old, new in paired) / len(paired)
        turns_delta = sum(new.turns - old.turns for old, new in paired) / len(paired)
        lines.append(
            f"- {label}: success delta {success_delta:+.3f}, reward delta {reward_delta:+.3f}, "
            f"turn delta {turns_delta:+.2f} over {len(paired)} paired games."
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def selected_variants(mode: str) -> list[str]:
    if mode == "direct":
        return ["controller_direct_base", "controller_direct_textgrad_rule"]
    if mode == "qwen":
        return ["qwen_ranker_base", "qwen_ranker_textgrad_rule"]
    return list(VARIANT_CONFIGS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/textworld24_structured_probe")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=62001)
    parser.add_argument("--mode", choices=["direct", "qwen", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = default_specs(args.seed)
    if args.limit:
        specs = specs[: args.limit]
    game_paths = ensure_games(specs, output_dir / "games")
    variants = selected_variants(args.mode)
    model = OpenAICompatibleChatModel(
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=16,
        timeout=120,
    )

    records: list[SuiteRecord] = []
    for spec_index, spec in enumerate(specs, start=1):
        for variant in variants:
            config = VARIANT_CONFIGS[variant]
            record = run_structured_variant(
                variant=variant,
                game_path=game_paths[spec.spec_id],
                model=model if config["use_qwen_ranker"] else None,
                use_qwen_ranker=config["use_qwen_ranker"],
                textgrad_rule=config["textgrad_rule"],
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
            suite = suite_record(spec=spec, record=record, model=args.model, temperature=args.temperature)
            records.append(suite)
            print(
                f"[{spec_index:02d}/{len(specs):02d}] {spec.spec_id} {variant}: "
                f"success={int(suite.success)} reward={suite.reward:.3f} turns={suite.turns} "
                f"qwen_calls={suite.qwen_calls} fallbacks={suite.fallback_calls}",
                flush=True,
            )

    aggregate = summarize_records(records)
    write_json(output_dir / "records.json", records)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_csv(output_dir / "per_problem.csv", per_problem_rows(records))
    write_markdown(output_dir / "summary.md", records, aggregate)
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
