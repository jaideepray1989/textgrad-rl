"""Build paper-facing success and step-efficiency evidence across text-game suites."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
QWEN_RUN = ROOT / "runs" / "qwen25_7b_full_textgames"
STRUCTURED_TW_RUN = ROOT / "runs" / "textworld24_structured_probe" / "records.json"
OUT_DIR = ROOT / "runs" / "realm_success_efficiency_evidence"
REPORT = ROOT / "REALM_SUCCESS_EFFICIENCY_EVIDENCE.md"


@dataclass(frozen=True)
class Episode:
    key: tuple[str, int]
    success: bool
    turns: int


@dataclass(frozen=True)
class BenchmarkPair:
    benchmark: str
    protocol: str
    baseline: tuple[Episode, ...]
    textgrad: tuple[Episode, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episodes(
    rows: Iterable[dict[str, Any]],
    *,
    id_field: str,
) -> tuple[Episode, ...]:
    result = tuple(
        Episode(
            key=(str(row[id_field]), int(row["seed"])),
            success=bool(row["success"]),
            turns=int(row["turns"]),
        )
        for row in rows
    )
    keys = [episode.key for episode in result]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate episode keys for {id_field}")
    return result


def load_textarena() -> BenchmarkPair:
    base_rows: list[dict[str, Any]] = []
    textgrad_rows: list[dict[str, Any]] = []
    for suite in ("puzzle_slm", "social_slm", "real_slm"):
        game_dir = QWEN_RUN / "textarena" / suite / "games"
        base_rows.extend(read_jsonl(game_dir / "fixed_prompt_slm_test.jsonl"))
        textgrad_rows.extend(read_jsonl(game_dir / "textgrad_rl_train_val_slm_test.jsonl"))
    return BenchmarkPair(
        benchmark="TextArena",
        protocol="Prompt-policy update",
        baseline=episodes(base_rows, id_field="env_id"),
        textgrad=episodes(textgrad_rows, id_field="env_id"),
    )


def load_textworld_express() -> BenchmarkPair:
    run_dir = QWEN_RUN / "textworld_slm" / "textworld_express"
    return BenchmarkPair(
        benchmark="TextWorldExpress",
        protocol="Prompt-policy update",
        baseline=episodes(read_jsonl(run_dir / "no_textgrad" / "test.jsonl"), id_field="problem_id"),
        textgrad=episodes(read_jsonl(run_dir / "textgrad_rl" / "test.jsonl"), id_field="problem_id"),
    )


def load_textworld() -> BenchmarkPair:
    if not STRUCTURED_TW_RUN.exists():
        raise FileNotFoundError(STRUCTURED_TW_RUN)
    rows = json.loads(STRUCTURED_TW_RUN.read_text(encoding="utf-8"))
    base = [row for row in rows if row["variant"] == "qwen_ranker_base"]
    textgrad = [row for row in rows if row["variant"] == "qwen_ranker_textgrad_rule"]
    return BenchmarkPair(
        benchmark="TextWorld",
        protocol="Structured controller-rule update",
        baseline=episodes(base, id_field="spec_id"),
        textgrad=episodes(textgrad, id_field="spec_id"),
    )


def summarize(pair: BenchmarkPair) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = {episode.key: episode for episode in pair.baseline}
    textgrad = {episode.key: episode for episode in pair.textgrad}
    if baseline.keys() != textgrad.keys():
        missing_textgrad = sorted(baseline.keys() - textgrad.keys())
        missing_baseline = sorted(textgrad.keys() - baseline.keys())
        raise ValueError(
            f"Mismatched pairs for {pair.benchmark}: "
            f"missing TextGrad={missing_textgrad}, missing baseline={missing_baseline}"
        )

    paired_rows: list[dict[str, Any]] = []
    common_step_decreases: list[int] = []
    for key in sorted(baseline):
        old = baseline[key]
        new = textgrad[key]
        common_success = old.success and new.success
        step_decrease = old.turns - new.turns if common_success else None
        if step_decrease is not None:
            common_step_decreases.append(step_decrease)
        paired_rows.append(
            {
                "benchmark": pair.benchmark,
                "problem": key[0],
                "seed": key[1],
                "baseline_success": int(old.success),
                "textgrad_success": int(new.success),
                "baseline_turns": old.turns,
                "textgrad_turns": new.turns,
                "common_success": int(common_success),
                "paired_step_decrease": "" if step_decrease is None else step_decrease,
            }
        )

    n = len(baseline)
    old_successes = sum(episode.success for episode in baseline.values())
    new_successes = sum(episode.success for episode in textgrad.values())
    success_delta = (new_successes - old_successes) / n
    return (
        {
            "benchmark": pair.benchmark,
            "protocol": pair.protocol,
            "n": n,
            "baseline_successes": old_successes,
            "baseline_success_rate": old_successes / n,
            "textgrad_successes": new_successes,
            "textgrad_success_rate": new_successes / n,
            "average_success_rate_improvement": success_delta,
            "common_successes": len(common_step_decreases),
            "mean_step_decrease_common_success": (
                statistics.mean(common_step_decreases) if common_step_decreases else None
            ),
            "median_step_decrease_common_success": (
                statistics.median(common_step_decreases) if common_step_decreases else None
            ),
        },
        paired_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.1f} pp"


def step(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def main() -> None:
    pairs = [load_textarena(), load_textworld_express(), load_textworld()]
    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for pair in pairs:
        summary, rows = summarize(pair)
        summaries.append(summary)
        paired_rows.extend(rows)

    total_n = sum(row["n"] for row in summaries)
    total_old = sum(row["baseline_successes"] for row in summaries)
    total_new = sum(row["textgrad_successes"] for row in summaries)
    all_common_steps = [
        float(row["paired_step_decrease"])
        for row in paired_rows
        if row["paired_step_decrease"] != ""
    ]
    pooled = {
        "benchmark": "Task-weighted aggregate",
        "protocol": "Mixed; descriptive only",
        "n": total_n,
        "baseline_successes": total_old,
        "baseline_success_rate": total_old / total_n,
        "textgrad_successes": total_new,
        "textgrad_success_rate": total_new / total_n,
        "average_success_rate_improvement": (total_new - total_old) / total_n,
        "common_successes": len(all_common_steps),
        "mean_step_decrease_common_success": statistics.mean(all_common_steps),
        "median_step_decrease_common_success": statistics.median(all_common_steps),
    }

    write_csv(OUT_DIR / "benchmark_summary.csv", summaries + [pooled])
    write_csv(OUT_DIR / "paired_episodes.csv", paired_rows)

    table_rows = summaries + [pooled]
    lines = [
        "# REALM Success and Step-Efficiency Evidence",
        "",
        "All rows use the local `qwen2.5:7b` actor at temperature `0.7`. "
        "The baseline is the same actor without TextGrad; the treatment applies the gated TextGrad-RL prompt policy or the structured controller-rule update.",
        "",
        "| Benchmark | Protocol | N | No TextGrad success | TextGrad-RL success | Avg. success improvement | Both solve | Mean step decrease | Median step decrease |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            "| {benchmark} | {protocol} | {n} | {old}/{n} ({old_rate}) | "
            "{new}/{n} ({new_rate}) | {delta} | {common} | {mean_steps} | {median_steps} |".format(
                benchmark=row["benchmark"],
                protocol=row["protocol"],
                n=row["n"],
                old=row["baseline_successes"],
                old_rate=pct(row["baseline_success_rate"]),
                new=row["textgrad_successes"],
                new_rate=pct(row["textgrad_success_rate"]),
                delta=signed_pp(row["average_success_rate_improvement"]),
                common=row["common_successes"],
                mean_steps=step(row["mean_step_decrease_common_success"]),
                median_steps=step(row["median_step_decrease_common_success"]),
            )
        )

    lines.extend(
        [
            "",
            "A positive step decrease means TextGrad-RL used fewer actions. Step efficiency is computed only on exactly paired test instances that both policies solved; this avoids treating a capped failure as an unusually long successful trajectory.",
            "",
            "## Interpretation",
            "",
            "- **TextArena:** success rises from 8.3% to 25.0% (+16.7 points). The sole baseline success is also solved by TextGrad-RL, but TextGrad-RL takes one extra action on that instance (mean and median decrease -1).",
            "- **TextWorldExpress:** both policies solve only Simon Says (12.5%), in five actions. The validation gate rejects the candidate update, so this benchmark provides a null result rather than evidence of improvement.",
            "- **TextWorld:** structured TextGrad controller rules raise Qwen-ranked success from 45.8% to 62.5% (+16.7 points). On the 11 common successes, mean action reduction is small and median reduction is zero; the main gain is four newly solved tasks, not faster execution on already-solved tasks.",
            "- **Aggregate:** task-weighted success rises from 29.5% to 43.2% (+13.6 points), but this is descriptive because TextWorld uses structured controller-rule optimization while the other suites use prompt-policy updates.",
            "- **Macro average:** averaging the three benchmark-level success rates gives 22.2% without TextGrad and 33.3% with TextGrad-RL, an average improvement of +11.1 percentage points.",
            "",
            "## Recommended Paper Claim",
            "",
            "> Across 44 locally executable held-out tasks with Qwen2.5-7B at temperature 0.7, TextGrad-RL increased task-weighted success from 29.5% to 43.2% (+13.6 percentage points). Gains appeared on TextArena and on TextWorld when textual gradients updated structured controller rules, while TextWorldExpress was unchanged. Among tasks solved by both methods, action-count efficiency was essentially unchanged (median reduction 0), indicating that the present benefit is improved task completion rather than shorter successful trajectories.",
            "",
            "## Publication Caveats",
            "",
            "These are pilot estimates with one test seed per environment/task, so they should not be presented as statistically conclusive. The strongest workshop version should repeat stochastic SLM evaluation with at least 10 seeds per task (ideally 30), report paired bootstrap confidence intervals, and keep the structured TextWorld protocol explicitly separated from direct prompt-policy optimization.",
            "",
            "## Sources",
            "",
            "- TextArena: `runs/qwen25_7b_full_textgames/textarena/*/games/*_test.jsonl` (12 tasks).",
            "- TextWorldExpress: `runs/qwen25_7b_full_textgames/textworld_slm/textworld_express/*/test.jsonl` (8 tasks).",
            "- TextWorld: `runs/textworld24_structured_probe/records.json`, Qwen-ranked variants (24 tasks, 80-action cap).",
            "- Machine-readable outputs: `runs/realm_success_efficiency_evidence/benchmark_summary.csv` and `paired_episodes.csv`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {OUT_DIR / 'benchmark_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'paired_episodes.csv'}")


if __name__ == "__main__":
    main()
