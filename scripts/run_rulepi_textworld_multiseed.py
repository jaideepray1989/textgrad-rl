"""Run and summarize the disjoint TextWorld-24 RulePI experiment across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


METHODS = [
    "fixed_prompt",
    "retry_with_diagnostics",
    "ungated_persistent_rules",
    "textgrad_policy_iteration",
]
COMPARISONS = [
    ("retry_with_diagnostics", "fixed_prompt"),
    ("ungated_persistent_rules", "fixed_prompt"),
    ("textgrad_policy_iteration", "fixed_prompt"),
    ("textgrad_policy_iteration", "retry_with_diagnostics"),
    ("textgrad_policy_iteration", "ungated_persistent_rules"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def hierarchical_bootstrap(
    grouped: dict[int, list[Any]],
    statistic: Callable[[list[Any]], float],
    *,
    rng: random.Random,
    samples: int,
) -> tuple[float, float, float]:
    observed = statistic([item for group in grouped.values() for item in group])
    outer_ids = sorted(grouped)
    estimates: list[float] = []
    for _ in range(samples):
        sample: list[Any] = []
        for _outer in outer_ids:
            chosen_outer = rng.choice(outer_ids)
            group = grouped[chosen_outer]
            sample.extend(rng.choice(group) for _item in group)
        estimates.append(statistic(sample))
    return observed, percentile(estimates, 0.025), percentile(estimates, 0.975)


def mean_field(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: sum(float(row[field]) for row in rows) / len(rows)


def paired_field(field: str) -> Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float]:
    return lambda pairs: sum(float(new[field]) - float(old[field]) for new, old in pairs) / len(pairs)


def run_seed(args: argparse.Namespace, repetition: int) -> Path:
    requested_seed = args.base_seed + repetition * args.seed_stride
    for fallback in range(args.max_seed_fallbacks + 1):
        seed = requested_seed + fallback
        suffix = "" if fallback == 0 else f"_fallback_{fallback:02d}"
        run_dir = Path(args.output_dir) / f"seed_{repetition:02d}{suffix}"
        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[{repetition + 1}/{args.repetitions}] reusing {run_dir}", flush=True)
            return run_dir
        command = [
            sys.executable,
            "-m",
            "textgrad_rl.benchmarks.textworld_24_suite",
            "--methods",
            ",".join(METHODS),
            "--seed",
            str(seed),
            "--max-steps",
            str(args.max_steps),
            "--min-mean-delta",
            str(args.min_mean_delta),
            "--output-dir",
            str(run_dir),
        ]
        print(
            f"[{repetition + 1}/{args.repetitions}] requested_seed={requested_seed} actual_seed={seed}",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return run_dir
        print(f"generation failed for seed={seed}; trying the next deterministic seed", flush=True)
    raise RuntimeError(
        f"Could not generate repetition {repetition} after {args.max_seed_fallbacks + 1} seeds"
    )


def summarize(args: argparse.Namespace, run_dirs: list[Path]) -> None:
    output_dir = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    for repetition, run_dir in enumerate(run_dirs):
        config = read_json(run_dir / "config.json")
        run_manifest.append(
            {
                "repetition": repetition,
                "requested_seed": args.base_seed + repetition * args.seed_stride,
                "actual_seed": int(config["test_specs"][0]["seed"]),
                "run_dir": str(run_dir),
            }
        )
        gate_map = read_json(run_dir / "gate_decisions.json")
        for method in METHODS:
            for record in read_jsonl(run_dir / method / "test.jsonl"):
                records.append({"repetition": repetition, **record})
            gates.append({"repetition": repetition, **gate_map[method]})

    rng = random.Random(args.bootstrap_seed)
    method_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_records = [row for row in records if row["method"] == method]
        grouped = {
            repetition: [row for row in method_records if row["repetition"] == repetition]
            for repetition in range(args.repetitions)
        }
        success, success_low, success_high = hierarchical_bootstrap(
            grouped, mean_field("success"), rng=rng, samples=args.bootstrap_samples
        )
        method_gates = [gate for gate in gates if gate["method"] == method]
        test_actions = sum(int(row["total_turns"]) for row in method_records)
        optimization_actions = sum(int(gate.get("optimization_environment_turns", 0)) for gate in method_gates)
        method_rows.append(
            {
                "method": method,
                "test_games": len(method_records),
                "success": success,
                "success_ci_low": success_low,
                "success_ci_high": success_high,
                "reward": mean_field("reward")(method_records),
                "attempts_per_game": mean_field("attempts")(method_records),
                "test_actions_per_game": test_actions / len(method_records),
                "optimization_actions": optimization_actions,
                "total_actions": test_actions + optimization_actions,
                "accepted_updates": sum(int(gate.get("accepted", False)) for gate in method_gates),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for new_method, old_method in COMPARISONS:
        grouped_pairs: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for repetition in range(args.repetitions):
            new = {
                row["spec_id"]: row
                for row in records
                if row["repetition"] == repetition and row["method"] == new_method
            }
            old = {
                row["spec_id"]: row
                for row in records
                if row["repetition"] == repetition and row["method"] == old_method
            }
            grouped_pairs[repetition] = [(new[key], old[key]) for key in sorted(new.keys() & old.keys())]
        for metric in ["success", "reward", "total_turns"]:
            point, low, high = hierarchical_bootstrap(
                grouped_pairs,
                paired_field(metric),
                rng=rng,
                samples=args.bootstrap_samples,
            )
            paired_rows.append(
                {
                    "comparison": f"{new_method} - {old_method}",
                    "metric": metric,
                    "mean_delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_games": sum(len(group) for group in grouped_pairs.values()),
                }
            )

    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "paired_bootstrap.csv", paired_rows)
    write_csv(output_dir / "all_test_records.csv", records)
    write_csv(output_dir / "gate_decisions.csv", gates)
    write_csv(output_dir / "run_manifest.csv", run_manifest)

    lines = [
        "# RulePI TextWorld-24 Multi-Seed Evidence",
        "",
        f"Generation seeds: {args.repetitions}; test games per seed: 24; horizon per attempt: {args.max_steps}.",
        f"Intervals: 95% hierarchical paired bootstrap ({args.bootstrap_samples:,} resamples).",
        "Training, validation, and test games use disjoint seeds within every repetition.",
        "Generator-invalid requested seeds are replaced by the next deterministic seed and logged in `run_manifest.csv`.",
        "",
        "## Methods",
        "",
        "| Method | Games | Success [95% CI] | Reward | Attempts/game | Test actions/game | Optimization actions | Total actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['test_games']} | {100 * row['success']:.1f}% "
            f"[{100 * row['success_ci_low']:.1f}, {100 * row['success_ci_high']:.1f}] | "
            f"{row['reward']:.3f} | {row['attempts_per_game']:.2f} | "
            f"{row['test_actions_per_game']:.2f} | {row['optimization_actions']} | {row['total_actions']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Effects",
            "",
            "| Comparison | Metric | Mean delta [95% CI] | Paired games |",
            "|---|---|---:|---:|",
        ]
    )
    for row in paired_rows:
        scale = 100 if row["metric"] == "success" else 1
        suffix = " pp" if row["metric"] == "success" else ""
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {scale * row['mean_delta']:+.2f}{suffix} "
            f"[{scale * row['ci_low']:+.2f}, {scale * row['ci_high']:+.2f}] | {row['paired_games']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=62001)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--min-mean-delta", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=44117)
    parser.add_argument("--max-seed-fallbacks", type=int, default=20)
    parser.add_argument("--output-dir", default="runs/rulepi_textworld_10seed")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_dirs = [run_seed(args, repetition) for repetition in range(args.repetitions)]
    summarize(args, run_dirs)
    print(f"RulePI multi-seed artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
