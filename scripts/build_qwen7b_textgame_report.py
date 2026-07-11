"""Build the qwen2.5:7b local text-game SLM report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "qwen25_7b_full_textgames"
REPORT = ROOT / "QWEN25_7B_TEXTGAME_REPORT.md"
OUT_DIR = RUN_DIR / "report_tables"


TEXTARENA_METHODS = {
    "fixed_prompt_slm": "no_textgrad",
    "textgrad_rl_train_val_slm": "textgrad_rl",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def i(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(float(value)) if value not in {"", None} else 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def load_textarena() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for suite_dir, label in [
        ("puzzle_slm", "TextArena puzzle"),
        ("social_slm", "TextArena social"),
        ("real_slm", "TextArena real_slm"),
    ]:
        path = RUN_DIR / "textarena" / suite_dir
        for row in read_csv(path / "metrics.csv"):
            if row.get("split") != "test" or row.get("method") not in TEXTARENA_METHODS:
                continue
            summaries.append(
                {
                    "benchmark": label,
                    "policy": TEXTARENA_METHODS[row["method"]],
                    "problems": {"puzzle_slm": 4, "social_slm": 3, "real_slm": 5}[suite_dir],
                    "episodes": i(row, "episodes"),
                    "reward": f(row, "average_reward"),
                    "score": f(row, "average_score"),
                    "success": f(row, "success_rate"),
                    "invalid": f(row, "invalid_move_rate"),
                    "parse_fail": 0.0,
                    "repeated": 0.0,
                    "truncated": f(row, "truncation_rate"),
                    "turns": f(row, "average_turns"),
                    "accepted_updates": 1 if row.get("accepted") == "True" else 0,
                    "gradient_count": "",
                    "max_steps": "env default",
                }
            )
        for row in read_csv(path / "per_env_metrics.csv"):
            if row.get("method") not in TEXTARENA_METHODS:
                continue
            problems.append(
                {
                    "benchmark": label,
                    "policy": TEXTARENA_METHODS[row["method"]],
                    "problem": row["env_id"],
                    "reward": f(row, "average_reward"),
                    "score": f(row, "average_score"),
                    "success": f(row, "success_rate"),
                    "invalid": f(row, "invalid_move_rate"),
                    "parse_fail": 0.0,
                    "repeated": 0.0,
                    "truncated": f(row, "truncation_rate"),
                    "turns": f(row, "average_turns"),
                }
            )
        for line in (path / "update_decisions.jsonl").read_text(encoding="utf-8").splitlines() if (path / "update_decisions.jsonl").exists() else []:
            if not line.strip():
                continue
            decision = json.loads(line)
            gates.append(
                {
                    "benchmark": label,
                    "policy": TEXTARENA_METHODS.get(decision.get("method", ""), decision.get("method", "")),
                    "accepted": decision.get("accepted", False),
                    "old_score": decision.get("old_score", 0.0),
                    "new_score": decision.get("new_score", 0.0),
                    "gradient_count": "",
                    "note": decision.get("gate_mode", ""),
                }
            )
    return summaries, problems, gates


def load_textworld_suite(name: str, label: str, max_steps: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = RUN_DIR / "textworld_slm" / name
    summaries: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for row in read_csv(path / "summary.csv"):
        summaries.append(
            {
                "benchmark": label,
                "policy": row["method"],
                "problems": i(row, "problems"),
                "episodes": i(row, "episodes"),
                "reward": f(row, "average_reward"),
                "score": f(row, "average_score"),
                "success": f(row, "success_rate"),
                "invalid": f(row, "invalid_action_rate"),
                "parse_fail": f(row, "parse_fail_rate"),
                "repeated": f(row, "repeated_action_rate"),
                "truncated": "",
                "turns": f(row, "average_turns"),
                "accepted_updates": i(row, "accepted_updates"),
                "gradient_count": i(row, "gradient_count"),
                "max_steps": max_steps,
            }
        )
    for row in read_csv(path / "slice_summary.csv"):
        if row.get("slice") != "problem_id":
            continue
        problems.append(
            {
                "benchmark": label,
                "policy": row["method"],
                "problem": row["value"],
                "reward": f(row, "average_reward"),
                "score": f(row, "average_score"),
                "success": f(row, "success_rate"),
                "invalid": f(row, "invalid_action_rate"),
                "parse_fail": f(row, "parse_fail_rate"),
                "repeated": f(row, "repeated_action_rate"),
                "truncated": "",
                "turns": f(row, "average_turns"),
            }
        )
    for policy, gate in read_json(path / "gate_decisions.json").items():
        gates.append(
            {
                "benchmark": label,
                "policy": policy,
                "accepted": gate.get("accepted", False),
                "old_score": gate.get("old_val_score", ""),
                "new_score": gate.get("new_val_score", ""),
                "gradient_count": gate.get("gradient_count", 0),
                "note": "",
            }
        )
    return summaries, problems, gates


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        by_policy = {row["policy"]: row for row in rows if row["benchmark"] == benchmark}
        if "no_textgrad" not in by_policy or "textgrad_rl" not in by_policy:
            continue
        base = by_policy["no_textgrad"]
        tg = by_policy["textgrad_rl"]
        out.append(
            {
                "benchmark": benchmark,
                "delta_reward": tg["reward"] - base["reward"],
                "delta_score": tg["score"] - base["score"],
                "delta_success": tg["success"] - base["success"],
                "delta_invalid": tg["invalid"] - base["invalid"],
                "delta_parse_fail": tg["parse_fail"] - base["parse_fail"],
                "delta_repeated": (tg["repeated"] or 0.0) - (base["repeated"] or 0.0),
                "delta_turns": tg["turns"] - base["turns"],
                "accepted_updates": tg["accepted_updates"],
            }
        )
    return out


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return lines


def main() -> None:
    ta_summary, ta_problem, ta_gates = load_textarena()
    twx_summary, twx_problem, twx_gates = load_textworld_suite("textworld_express", "TextWorldExpress 8", "12")
    tw_summary, tw_problem, tw_gates = load_textworld_suite("textworld_24", "TextWorld 24", "3")
    summaries = ta_summary + twx_summary + tw_summary
    problems = ta_problem + twx_problem + tw_problem
    gates = ta_gates + twx_gates + tw_gates
    deltas = paired_deltas(summaries)

    write_csv(OUT_DIR / "summary.csv", summaries)
    write_csv(OUT_DIR / "paired_deltas.csv", deltas)
    write_csv(OUT_DIR / "per_problem.csv", problems)
    write_csv(OUT_DIR / "gate_decisions.csv", gates)

    lines = [
        "# qwen2.5:7b Local Text-Game SLM Report",
        "",
        "Model: `qwen2.5:7b` through the local Ollama OpenAI-compatible endpoint.",
        "Temperature: `0.7`.",
        "",
        "Policies:",
        "- `no_textgrad`: frozen SLM actor with the initial prompt-policy text only.",
        "- `textgrad_rl`: one TextGrad-RL textual rule update selected by a train/validation gate.",
        "",
        "Coverage:",
        "- TextArena: 12 SLM-enabled environments: 4 puzzle, 3 social, 5 real-SLM; one train/validation/test seed per environment.",
        "- TextWorldExpress: all 8 local games; one train/validation/test seed per game; 12-step cap.",
        "- TextWorld 24: all 24 generated Microsoft TextWorld games; 3-step cap because longer qwen2.5:7b TextWorld episodes produced long local model calls. Treat this as a shallow interaction probe, not a full solve-rate run.",
        "",
        "## Overall Results",
        "",
    ]
    lines.extend(
        markdown_table(
            summaries,
            [
                "benchmark",
                "policy",
                "problems",
                "episodes",
                "reward",
                "score",
                "success",
                "invalid",
                "parse_fail",
                "repeated",
                "truncated",
                "turns",
                "accepted_updates",
                "gradient_count",
                "max_steps",
            ],
        )
    )
    lines.extend(["", "## TextGrad-RL Minus No-TextGrad", ""])
    lines.extend(
        markdown_table(
            deltas,
            [
                "benchmark",
                "delta_reward",
                "delta_score",
                "delta_success",
                "delta_invalid",
                "delta_parse_fail",
                "delta_repeated",
                "delta_turns",
                "accepted_updates",
            ],
        )
    )
    lines.extend(["", "## Gate Decisions", ""])
    lines.extend(markdown_table(gates, ["benchmark", "policy", "accepted", "old_score", "new_score", "gradient_count", "note"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The clearest positive result is TextArena real-SLM: TextGrad-RL improves reward from 0.220 to 0.760, success from 0.200 to 0.600, and removes invalid moves in this one-seed sweep.",
            "- TextArena puzzle shows a small reward increase but worse invalid-action rate, so the update is not a clean win there.",
            "- TextArena social is unchanged: the qwen actor produces valid but non-winning long social trajectories, and the TextGrad rule does not change held-out outcomes.",
            "- TextWorldExpress and TextWorld 24 do not show meaningful improvement from the gated update in this run. Their gates reject the update, so the final policies are effectively the fixed prompt; small score differences come from stochastic qwen rollouts.",
            "- TextWorld 24 should be interpreted cautiously because the run uses a 3-step cap to complete all 24 games locally. A stronger claim needs either a per-call hard timeout/retry wrapper or a faster deterministic local model-serving path for longer episodes.",
            "",
            "## Artifacts",
            "",
            "- TextArena: `runs/qwen25_7b_full_textgames/textarena/`",
            "- TextWorldExpress/TextWorld: `runs/qwen25_7b_full_textgames/textworld_slm/`",
            "- Consolidated CSVs: `runs/qwen25_7b_full_textgames/report_tables/`",
        ]
    )
    REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote CSV tables under {OUT_DIR}")


if __name__ == "__main__":
    main()
