"""Build workshop-facing local experimental result tables.

The script only reads checked-in/local run artifacts and writes compact CSV/MD
summaries for the REALM workshop draft:

- bootstrap confidence intervals for aggregate metrics,
- paired fixed-vs-TextGrad deltas where episode keys align,
- per-family/per-environment breakdowns,
- learned rule and gate-decision examples,
- optional SLM temperature and candidate-count tables when those runs exist.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "workshop_local_experiments"
REPORT = ROOT / "WORKSHOP_LOCAL_EXPERIMENTS.md"
BOOTSTRAPS = 2000
RNG_SEED = 20260708


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def b(value: Any) -> bool:
    return bool(value)


def f(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def normalize_method(method: str) -> str:
    return {
        "fixed_actor": "fixed_prompt",
        "fixed_policy": "fixed_prompt",
        "fixed_prompt_slm": "fixed_prompt",
        "textgrad_rl": "textgrad_policy_iteration",
        "textgrad_policy_iteration_slm": "textgrad_policy_iteration",
        "textgrad_rl_no_gate_slm": "textgrad_rl_no_gate",
        "textgrad_rl_train_val_slm": "textgrad_rl_train_val",
    }.get(method, method)


def record_metric(record: dict[str, Any], metric: str) -> float:
    if metric == "reward":
        return f(record.get("reward", record.get("avg_reward", 0.0)))
    if metric == "score":
        return f(record.get("score", record.get("external_score", record.get("reward", 0.0))))
    if metric == "success":
        return 1.0 if b(record.get("success", False)) else 0.0
    if metric == "invalid":
        return 1.0 if b(record.get("invalid_move", record.get("invalid_action", record.get("invalid_browser_action", False)))) else 0.0
    if metric == "repeated":
        return 1.0 if b(record.get("repeated_actions", record.get("repeated_action", False))) else 0.0
    if metric == "truncated":
        return 1.0 if b(record.get("truncated", False)) else 0.0
    if metric == "turns":
        return f(record.get("turns", record.get("avg_turns", 0.0)))
    raise KeyError(metric)


def bootstrap_ci(values: list[float], iterations: int = BOOTSTRAPS) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0], values[0])
    rng = random.Random(RNG_SEED + len(values))
    samples = []
    n = len(values)
    for _ in range(iterations):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    lo = samples[int(0.025 * iterations)]
    hi = samples[int(0.975 * iterations) - 1]
    return (mean(values), lo, hi)


def episode_key(record: dict[str, Any]) -> tuple[Any, ...]:
    keys = [
        "suite",
        "env_id",
        "game",
        "spec_id",
        "task_id",
        "seed",
        "target_player",
        "target_side",
        "category",
    ]
    return tuple(record.get(key) for key in keys if key in record)


def collect_records() -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}

    broad = []
    for path in [
        ROOT / "runs/textarena_broad_50/fixed_prompt_episodes.jsonl",
        ROOT / "runs/textarena_broad_50/textgrad_policy_iteration_episodes.jsonl",
    ]:
        for row in read_jsonl(path):
            row = dict(row)
            row["method"] = normalize_method(str(row.get("method", "")))
            broad.append(row)
    datasets["TextArena Broad 50"] = broad

    difficulty = []
    diff_dir = ROOT / "runs/textarena_expanded_suites/difficulty_generalization/games"
    for method in ["fixed_prompt", "textgrad_policy_iteration"]:
        for path in sorted(diff_dir.glob(f"{method}_rep_*_difficulty_test.jsonl")):
            for row in read_jsonl(path):
                row = dict(row)
                row["method"] = method
                difficulty.append(row)
    datasets["TextArena Difficulty Generalization"] = difficulty

    miniwob = []
    for row in read_jsonl(ROOT / "runs/miniwob_subset_50x3/episodes.jsonl"):
        if row.get("split") != "test":
            continue
        method = normalize_method(str(row.get("method", "")))
        if method in {"fixed_prompt", "textgrad_policy_iteration"}:
            row = dict(row)
            row["method"] = method
            miniwob.append(row)
    datasets["BrowserGym MiniWoB++ 50"] = miniwob

    for name, base in [
        ("TextWorldExpress 8", ROOT / "runs/textworld_express_suite"),
        ("TextWorld 24", ROOT / "runs/textworld_24_suite"),
    ]:
        rows = []
        for method in ["fixed_prompt", "textgrad_policy_iteration"]:
            for row in read_jsonl(base / method / "test.jsonl"):
                row = dict(row)
                row["method"] = method
                rows.append(row)
        datasets[name] = rows

    transfer = []
    for suite in ["browser_transfer", "tau_transfer", "swe_transfer"]:
        for method, rel in [
            ("fixed_prompt", "fixed_test.jsonl"),
            ("textgrad_policy_iteration", "textgrad_test.jsonl"),
        ]:
            for row in read_jsonl(ROOT / "runs/external_transfer_protocols" / suite / rel):
                row = dict(row)
                row["suite"] = suite
                row["method"] = method
                transfer.append(row)
    datasets["Local Transfer Protocol Probe"] = transfer

    slm = []
    for suite in ["puzzle_slm", "social_slm", "real_slm"]:
        games = ROOT / "runs/textarena_slm_candidate_pool_30seed" / suite / "games"
        for method in ["fixed_prompt_slm", "textgrad_rl_no_gate_slm", "textgrad_rl_train_val_slm"]:
            for row in read_jsonl(games / f"{method}_test.jsonl"):
                row = dict(row)
                row["method"] = normalize_method(method)
                row["suite"] = row.get("suite") or suite.replace("_slm", "")
                slm.append(row)
    datasets["TextArena SLM Candidate Pool 30-Seed"] = slm

    temp_t00 = []
    for suite in ["puzzle_slm", "social_slm", "real_slm"]:
        games = ROOT / "runs/textarena_slm_qwen25_3b_t00_30seed_fixed" / suite / "games"
        for row in read_jsonl(games / "fixed_prompt_slm_test.jsonl"):
            row = dict(row)
            row["method"] = "fixed_prompt"
            row["temperature"] = "0.0"
            row["suite"] = row.get("suite") or suite.replace("_slm", "")
            temp_t00.append(row)
    if temp_t00:
        datasets["TextArena SLM Fixed Temperature 0.0"] = temp_t00

    return datasets


def aggregate_rows(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    metrics = ["reward", "success", "invalid", "repeated", "truncated", "turns"]
    for benchmark, rows in datasets.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("method", ""))].append(row)
        for method, items in sorted(grouped.items()):
            out: dict[str, Any] = {"benchmark": benchmark, "method": method, "episodes": len(items)}
            for metric in metrics:
                vals = [record_metric(item, metric) for item in items]
                avg, lo, hi = bootstrap_ci(vals)
                out[metric] = round(avg, 4)
                out[f"{metric}_ci_low"] = round(lo, 4)
                out[f"{metric}_ci_high"] = round(hi, 4)
            output.append(out)
    return output


def paired_delta_rows(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    metrics = ["reward", "success", "invalid", "repeated", "truncated", "turns"]
    for benchmark, rows in datasets.items():
        by_method_key: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
        for row in rows:
            by_method_key[(str(row.get("method", "")), episode_key(row))] = row
        fixed_keys = {key for method, key in by_method_key if method == "fixed_prompt"}
        methods = sorted({method for method, _ in by_method_key if method != "fixed_prompt"})
        for method in methods:
            keys = sorted(fixed_keys & {key for m, key in by_method_key if m == method})
            if not keys:
                continue
            out: dict[str, Any] = {"benchmark": benchmark, "comparison": f"{method} - fixed_prompt", "paired_episodes": len(keys)}
            for metric in metrics:
                deltas = [
                    record_metric(by_method_key[(method, key)], metric) - record_metric(by_method_key[("fixed_prompt", key)], metric)
                    for key in keys
                ]
                avg, lo, hi = bootstrap_ci(deltas)
                out[f"{metric}_delta"] = round(avg, 4)
                out[f"{metric}_delta_ci_low"] = round(lo, 4)
                out[f"{metric}_delta_ci_high"] = round(hi, 4)
            output.append(out)
    return output


def copy_breakdowns() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("TextArena Broad 50 per-env", ROOT / "runs/textarena_broad_50/per_env_metrics.csv"),
        ("TextArena Broad 50 support slice", ROOT / "runs/textarena_broad_50/slice_summary.csv"),
        ("MiniWoB++ category", ROOT / "runs/miniwob_subset_50x3/category_summary.csv"),
        ("TextWorldExpress game", ROOT / "runs/textworld_express_suite/slice_summary.csv"),
        ("TextWorld 24 family", ROOT / "runs/textworld_24_suite/slice_summary.csv"),
    ]
    for label, path in sources:
        for row in read_csv(path):
            method = normalize_method(row.get("method", ""))
            if "ppo" in method:
                continue
            row = dict(row)
            row["breakdown"] = label
            row["method"] = method
            rows.append(row)
    return rows


def rule_lines_from_text_variables(path: Path) -> list[str]:
    data = read_json(path)
    lines: list[str] = []
    for variable in data.values():
        value = variable.get("value", "") if isinstance(variable, dict) else ""
        for line in value.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append(stripped[2:])
    return lines


def learned_rules_and_gates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in read_jsonl(ROOT / "runs/miniwob_subset_50x3/prompt_updates.jsonl"):
        if item.get("method") == "textgrad_rl":
            prompt = item.get("candidate_prompt", {}).get("general_agent_policy", "")
            rules = [line.strip()[2:] for line in prompt.splitlines() if line.strip().startswith("- ")]
            rows.append(
                {
                    "benchmark": "BrowserGym MiniWoB++ 50",
                    "method": "textgrad_policy_iteration",
                    "accepted": item.get("accepted"),
                    "old_score": item.get("gate_details", {}).get("old_score"),
                    "new_score": item.get("gate_details", {}).get("new_score"),
                    "learned_rules": " | ".join(rules),
                }
            )

    for benchmark, base in [
        ("TextWorldExpress 8", ROOT / "runs/textworld_express_suite"),
        ("TextWorld 24", ROOT / "runs/textworld_24_suite"),
    ]:
        gate = read_json(base / "textgrad_policy_iteration/gate_decision.json")
        rows.append(
            {
                "benchmark": benchmark,
                "method": "textgrad_policy_iteration",
                "accepted": gate.get("accepted"),
                "old_score": gate.get("old_val_score"),
                "new_score": gate.get("new_val_score"),
                "learned_rules": " | ".join(rule_lines_from_text_variables(base / "textgrad_policy_iteration/text_variables.json")[:8]),
            }
        )

    for suite in ["browser_transfer", "tau_transfer", "swe_transfer"]:
        gate = read_json(ROOT / "runs/external_transfer_protocols" / suite / "update_decision.json")
        rows.append(
            {
                "benchmark": f"Local Transfer Protocol Probe/{suite}",
                "method": "textgrad_policy_iteration",
                "accepted": gate.get("accepted"),
                "old_score": gate.get("old_train_score"),
                "new_score": gate.get("new_train_score"),
                "learned_rules": " | ".join(gate.get("learned_rule_ids", [])),
            }
        )

    for suite in ["puzzle_slm", "social_slm", "real_slm"]:
        for item in read_jsonl(ROOT / "runs/textarena_slm_candidate_pool_30seed" / suite / "update_decisions.jsonl"):
            method = normalize_method(str(item.get("method", "")))
            if method not in {"textgrad_rl_no_gate", "textgrad_rl_train_val"}:
                continue
            rows.append(
                {
                    "benchmark": f"TextArena SLM 30-Seed/{suite.replace('_slm', '')}",
                    "method": method,
                    "accepted": item.get("accepted"),
                    "old_score": item.get("old_score"),
                    "new_score": item.get("new_score"),
                    "learned_rules": item.get("candidate_description"),
                }
            )

    return rows


def temperature_rows(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    t07_records = [
        row
        for row in datasets.get("TextArena SLM Candidate Pool 30-Seed", [])
        if row.get("method") == "fixed_prompt"
    ]
    t00_records = datasets.get("TextArena SLM Fixed Temperature 0.0", [])
    for label, records in [("0.0", t00_records), ("0.7", t07_records)]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[str(row.get("suite", "unknown"))].append(row)
        for suite, items in sorted(grouped.items()):
            out = {"suite": suite, "temperature": label, "episodes": len(items)}
            for metric in ["reward", "success", "invalid", "truncated", "turns"]:
                vals = [record_metric(item, metric) for item in items]
                avg, lo, hi = bootstrap_ci(vals)
                out[metric] = round(avg, 4)
                out[f"{metric}_ci_low"] = round(lo, 4)
                out[f"{metric}_ci_high"] = round(hi, 4)
            rows.append(out)
    return rows


def candidate_count_rows() -> list[dict[str, Any]]:
    rows = []
    # Full held-out data currently exists for candidate_count=8. Candidate-count
    # runs for 1 and 4 are written by the companion shell script when launched.
    for count, root in [
        (1, ROOT / "runs/textarena_slm_candidate_count_1_30seed"),
        (4, ROOT / "runs/textarena_slm_candidate_count_4_30seed"),
        (8, ROOT / "runs/textarena_slm_candidate_pool_30seed"),
    ]:
        for suite in ["puzzle_slm", "social_slm", "real_slm"]:
            metrics = read_csv(root / suite / "metrics.csv")
            for row in metrics:
                if row.get("split") != "test":
                    continue
                method = normalize_method(row.get("method", ""))
                if method not in {"fixed_prompt", "textgrad_rl_no_gate", "textgrad_rl_train_val"}:
                    continue
                rows.append(
                    {
                        "candidate_count": count,
                        "suite": row.get("suite", suite.replace("_slm", "")),
                        "method": method,
                        "episodes": row.get("episodes"),
                        "reward": round(f(row.get("average_reward")), 4),
                        "score": round(f(row.get("average_score")), 4),
                        "success": round(f(row.get("success_rate")), 4),
                        "invalid": round(f(row.get("invalid_move_rate")), 4),
                        "truncated": round(f(row.get("truncation_rate")), 4),
                        "turns": round(f(row.get("average_turns")), 4),
                    }
                )
    return rows


def candidate_index(candidate_id: str) -> int:
    match = re.match(r"candidate_(\d+)_", candidate_id)
    if not match:
        return 0
    return int(match.group(1))


def candidate_count_validation_rows() -> list[dict[str, Any]]:
    rows = []
    root = ROOT / "runs/textarena_slm_candidate_pool_30seed"
    for suite in ["puzzle_slm", "social_slm", "real_slm"]:
        decisions = read_jsonl(root / suite / "candidate_decisions.jsonl")
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for decision in decisions:
            method = normalize_method(str(decision.get("method", "")))
            if method in {"textgrad_rl_no_gate", "textgrad_rl_train_val"}:
                by_method[method].append(decision)
        for method, method_decisions in sorted(by_method.items()):
            for count in [1, 4, 8]:
                eligible = [
                    decision
                    for decision in method_decisions
                    if candidate_index(str(decision.get("candidate_id", ""))) < count
                ]
                if not eligible:
                    continue
                selected = max(
                    eligible,
                    key=lambda decision: (
                        1 if decision.get("accepted") else 0,
                        f(decision.get("candidate_rank_score")),
                    ),
                )
                metrics = selected.get("new_metrics", {})
                old = selected.get("old_metrics", {})
                rows.append(
                    {
                        "candidate_count": count,
                        "suite": suite.replace("_slm", ""),
                        "method": method,
                        "selected_candidate": selected.get("candidate_id"),
                        "accepted": selected.get("accepted"),
                        "candidate_rank_score": round(f(selected.get("candidate_rank_score")), 4),
                        "old_val_score": round(f(selected.get("old_score")), 4),
                        "selected_val_score": round(f(selected.get("new_score")), 4),
                        "val_score_delta": round(f(selected.get("new_score")) - f(selected.get("old_score")), 4),
                        "old_success": round(f(old.get("success_rate")), 4),
                        "selected_success": round(f(metrics.get("success_rate")), 4),
                        "old_invalid": round(f(old.get("invalid_move_rate")), 4),
                        "selected_invalid": round(f(metrics.get("invalid_move_rate")), 4),
                        "description": selected.get("candidate_description"),
                    }
                )
    return rows


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    shown = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in shown:
        lines.append("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    aggregate: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    breakdowns: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    temps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_validation: list[dict[str, Any]],
) -> None:
    lines = [
        "# Workshop Local Experiments",
        "",
        "Generated by `scripts/build_workshop_experimental_package.py` from local run artifacts.",
        "",
        "## Bootstrap Aggregate CIs",
        "",
        md_table(
            aggregate,
            ["benchmark", "method", "episodes", "reward", "reward_ci_low", "reward_ci_high", "success", "success_ci_low", "success_ci_high", "invalid", "invalid_ci_low", "invalid_ci_high"],
        ),
        "",
        "## Paired Fixed-vs-TextGrad Deltas",
        "",
        md_table(
            paired,
            ["benchmark", "comparison", "paired_episodes", "reward_delta", "reward_delta_ci_low", "reward_delta_ci_high", "success_delta", "success_delta_ci_low", "success_delta_ci_high", "invalid_delta", "invalid_delta_ci_low", "invalid_delta_ci_high"],
        ),
        "",
        "## Per-Family / Per-Environment Breakdown Files",
        "",
        "- `runs/workshop_local_experiments/breakdowns.csv` contains TextArena per-env/slice, MiniWoB category, TextWorldExpress game, and TextWorld family breakdowns.",
        "",
        "## Learned Rules and Gate Decisions",
        "",
        md_table(rules, ["benchmark", "method", "accepted", "old_score", "new_score", "learned_rules"], limit=20),
        "",
        "## Temperature Sensitivity",
        "",
        "Temperature rows appear when `runs/textarena_slm_qwen25_3b_t00_30seed_fixed` is available. The `t=0.7` rows are from the existing 30-seed candidate-pool run.",
        "",
        md_table(temps, ["suite", "temperature", "episodes", "reward", "success", "invalid", "truncated", "turns"]),
        "",
        "## Candidate-Count Ablation",
        "",
        "Candidate-count rows are included for any available 30-seed runs. The existing `candidate_count=8` run is always used; counts 1 and 4 are added after the companion run script finishes.",
        "",
        md_table(candidates, ["candidate_count", "suite", "method", "episodes", "reward", "score", "success", "invalid", "truncated", "turns"]),
        "",
        "## Candidate-Count Validation Replay",
        "",
        "This retrospective replay uses the existing 30-seed `candidate_count=8` run. For each method and suite, it replays the runner's selection rule over the first 1, first 4, and all 8 candidates, then reports the selected validation candidate. This is a candidate-diversity diagnostic; full held-out count-1/count-4 reruns can be launched with `scripts/run_slm_temperature_and_candidate_ablation.sh`.",
        "",
        md_table(
            candidate_validation,
            ["candidate_count", "suite", "method", "selected_candidate", "accepted", "old_val_score", "selected_val_score", "val_score_delta", "old_success", "selected_success", "old_invalid", "selected_invalid"],
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = collect_records()
    aggregate = aggregate_rows(datasets)
    paired = paired_delta_rows(datasets)
    breakdowns = copy_breakdowns()
    rules = learned_rules_and_gates()
    temps = temperature_rows(datasets)
    candidates = candidate_count_rows()
    candidate_validation = candidate_count_validation_rows()

    write_csv(OUT / "bootstrap_aggregate_cis.csv", aggregate)
    write_csv(OUT / "paired_deltas.csv", paired)
    write_csv(OUT / "breakdowns.csv", breakdowns)
    write_csv(OUT / "learned_rules_and_gates.csv", rules)
    write_csv(OUT / "temperature_sensitivity.csv", temps)
    write_csv(OUT / "candidate_count_ablation.csv", candidates)
    write_csv(OUT / "candidate_count_validation_replay.csv", candidate_validation)
    write_report(aggregate, paired, breakdowns, rules, temps, candidates, candidate_validation)
    print(f"Wrote {REPORT} and CSVs under {OUT}")


if __name__ == "__main__":
    main()
