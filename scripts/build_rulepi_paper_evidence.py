"""Compile TextWorld, TextArena, and Qwen boundary evidence for the RulePI paper."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
TEXTWORLD_DIR = ROOT / "runs" / "rulepi_textworld_10seed"
TEXTARENA_DIR = ROOT / "runs" / "rulepi_textarena_supported_10seed"
QWEN_DIR = ROOT / "runs" / "qwen25_7b_textworld24_full80_t5" / "textworld_24"
OUTPUT_DIR = ROOT / "runs" / "rulepi_paper_evidence"
REPORT = ROOT / "RULEPI_PAPER_EVIDENCE.md"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def bootstrap(
    grouped: dict[str, list[Any]],
    statistic: Callable[[list[Any]], float],
    rng: random.Random,
    samples: int = 10_000,
) -> tuple[float, float, float]:
    observed = statistic([item for group in grouped.values() for item in group])
    clusters = sorted(grouped)
    estimates: list[float] = []
    for _ in range(samples):
        sample: list[Any] = []
        for _cluster in clusters:
            group = grouped[rng.choice(clusters)]
            sample.extend(rng.choice(group) for _item in group)
        estimates.append(statistic(sample))
    return observed, percentile(estimates, 0.025), percentile(estimates, 0.975)


def mean_field(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: sum(float(row[field]) for row in rows) / len(rows)


def paired_field(field: str) -> Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float]:
    return lambda pairs: sum(float(new[field]) - float(old[field]) for new, old in pairs) / len(pairs)


def textarena_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = {
        method: read_jsonl(TEXTARENA_DIR / f"{method}_episodes.jsonl")
        for method in METHODS
    }
    rng = random.Random(77231)
    run_summary = {
        row["method"]: row for row in read_csv(TEXTARENA_DIR / "summary.csv")
    }
    summaries: list[dict[str, Any]] = []
    for method, rows in records.items():
        grouped = {
            env_id: [row for row in rows if row["env_id"] == env_id]
            for env_id in sorted({row["env_id"] for row in rows})
        }
        success, low, high = bootstrap(grouped, mean_field("success"), rng)
        summaries.append(
            {
                "method": method,
                "episodes": len(rows),
                "success": success,
                "success_ci_low": low,
                "success_ci_high": high,
                "reward": mean_field("reward")(rows),
                "attempts_per_episode": mean_field("attempts")(rows),
                "test_actions_per_episode": mean_field("total_turns")(rows),
                "optimization_actions": int(run_summary[method]["optimization_actions"]),
            }
        )

    paired: list[dict[str, Any]] = []
    for new_method, old_method in COMPARISONS:
        new = {
            (row["env_id"], row["seed"], row["target_side"]): row
            for row in records[new_method]
        }
        old = {
            (row["env_id"], row["seed"], row["target_side"]): row
            for row in records[old_method]
        }
        pairs = [(new[key], old[key]) for key in sorted(new.keys() & old.keys(), key=str)]
        grouped = {
            env_id: [pair for pair in pairs if pair[0]["env_id"] == env_id]
            for env_id in sorted({pair[0]["env_id"] for pair in pairs})
        }
        for metric in ["success", "reward", "total_turns"]:
            point, low, high = bootstrap(grouped, paired_field(metric), rng)
            paired.append(
                {
                    "comparison": f"{new_method} - {old_method}",
                    "metric": metric,
                    "mean_delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_episodes": len(pairs),
                }
            )
    return summaries, paired


def effect_row(rows: list[dict[str, str]], comparison: str, metric: str) -> dict[str, str]:
    return next(row for row in rows if row["comparison"] == comparison and row["metric"] == metric)


def main() -> None:
    textworld = read_csv(TEXTWORLD_DIR / "method_summary.csv")
    textworld_pairs = read_csv(TEXTWORLD_DIR / "paired_bootstrap.csv")
    textarena, textarena_pairs = textarena_evidence()
    qwen = {row["method"]: row for row in read_csv(QWEN_DIR / "summary.csv")}
    qwen_gate = json.loads((QWEN_DIR / "textgrad_rl" / "gate_decision.json").read_text(encoding="utf-8"))
    write_csv(OUTPUT_DIR / "textarena_method_summary.csv", textarena)
    write_csv(OUTPUT_DIR / "textarena_paired_bootstrap.csv", textarena_pairs)

    tw_effect = effect_row(
        textworld_pairs,
        "textgrad_policy_iteration - fixed_prompt",
        "success",
    )
    ta_effect = next(
        row
        for row in textarena_pairs
        if row["comparison"] == "textgrad_policy_iteration - fixed_prompt" and row["metric"] == "success"
    )
    lines = [
        "# RulePI Paper Evidence",
        "",
        "## Claim 1: Persistent textual rules improve controlled interactive agents",
        "",
        "| Benchmark | Fixed | Diagnostic retry | Ungated persistence | RulePI | RulePI - fixed (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tw_by_method = {row["method"]: row for row in textworld}
    ta_by_method = {row["method"]: row for row in textarena}
    lines.append(
        "| TextWorld-24, 10 generations | "
        f"{100 * float(tw_by_method['fixed_prompt']['success']):.1f}% | "
        f"{100 * float(tw_by_method['retry_with_diagnostics']['success']):.1f}% | "
        f"{100 * float(tw_by_method['ungated_persistent_rules']['success']):.1f}% | "
        f"{100 * float(tw_by_method['textgrad_policy_iteration']['success']):.1f}% | "
        f"{100 * float(tw_effect['mean_delta']):+.1f} pp "
        f"[{100 * float(tw_effect['ci_low']):+.1f}, {100 * float(tw_effect['ci_high']):+.1f}] |"
    )
    lines.append(
        "| TextArena supported-10, 10 seeds | "
        f"{100 * ta_by_method['fixed_prompt']['success']:.1f}% | "
        f"{100 * ta_by_method['retry_with_diagnostics']['success']:.1f}% | "
        f"{100 * ta_by_method['ungated_persistent_rules']['success']:.1f}% | "
        f"{100 * ta_by_method['textgrad_policy_iteration']['success']:.1f}% | "
        f"{100 * ta_effect['mean_delta']:+.1f} pp "
        f"[{100 * ta_effect['ci_low']:+.1f}, {100 * ta_effect['ci_high']:+.1f}] |"
    )
    lines.extend(
        [
            "",
            "## Claim 2: Persistence amortizes task-local diagnosis",
            "",
            "| Benchmark | Retry attempts | RulePI attempts | Retry test actions | RulePI test actions | RulePI optimization | Break-even tasks |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| TextWorld-24 | "
            f"{float(tw_by_method['retry_with_diagnostics']['attempts_per_game']):.2f} | "
            f"{float(tw_by_method['textgrad_policy_iteration']['attempts_per_game']):.2f} | "
            f"{float(tw_by_method['retry_with_diagnostics']['test_actions_per_game']):.2f} | "
            f"{float(tw_by_method['textgrad_policy_iteration']['test_actions_per_game']):.2f} | "
            f"{int(tw_by_method['textgrad_policy_iteration']['optimization_actions'])} | "
            f"{int(float(tw_by_method['textgrad_policy_iteration']['optimization_actions']) / (float(tw_by_method['retry_with_diagnostics']['test_actions_per_game']) - float(tw_by_method['textgrad_policy_iteration']['test_actions_per_game'])) + 0.999)} |",
            "| TextArena supported-10 | "
            f"{ta_by_method['retry_with_diagnostics']['attempts_per_episode']:.2f} | "
            f"{ta_by_method['textgrad_policy_iteration']['attempts_per_episode']:.2f} | "
            f"{ta_by_method['retry_with_diagnostics']['test_actions_per_episode']:.2f} | "
            f"{ta_by_method['textgrad_policy_iteration']['test_actions_per_episode']:.2f} | "
            f"{ta_by_method['textgrad_policy_iteration']['optimization_actions']} | "
            f"{int(ta_by_method['textgrad_policy_iteration']['optimization_actions'] / (ta_by_method['retry_with_diagnostics']['test_actions_per_episode'] - ta_by_method['textgrad_policy_iteration']['test_actions_per_episode']) + 0.999)} |",
            "",
            "RulePI crosses the interaction break-even point within the 240-game TextWorld study (about 202 deployments), but not within the 130-episode TextArena study (about 756 deployments). The efficiency claim is therefore deployment-time amortization, not uniformly lower total experimental compute.",
            "",
            "## Ablation and Boundary",
            "",
            "- Gated and ungated persistence have identical TextWorld success (56.7%) and test actions (42.25/game).",
            "- On TextArena they also tie in success (61.5%); RulePI has slightly higher reward (0.573 vs 0.564). These runs do not establish a validation-gate advantage.",
            f"- With qwen2.5:7b at temperature 0.7, the no-TextGrad and RulePI protocols both score {100 * float(qwen['no_textgrad']['success_rate']):.1f}% success. The gate rejects the edit because validation score moves from {qwen_gate['old_val_score']:.3f} to {qwen_gate['new_val_score']:.3f}, so the deployed RulePI policy remains the base policy.",
            "",
            "The supported positive claim is therefore about persistent textual rules for prompt-aware structured actors, not generic improvement of small language-model agents and not a demonstrated benefit from validation gating.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
