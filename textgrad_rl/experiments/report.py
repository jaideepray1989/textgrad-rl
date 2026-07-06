"""Generate Markdown summaries for experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from textgrad_rl.utils.json_utils import read_json


def generate_summary_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    environment = read_json(run_dir / "environment_info.json") if (run_dir / "environment_info.json").exists() else {}
    final_vars = read_json(run_dir / "final_text_variables.json") if (run_dir / "final_text_variables.json").exists() else {}
    rows = _read_metrics(run_dir / "metrics.csv")
    gradients = _read_gradients(run_dir / "gradients")
    accepted = _read_jsonl(run_dir / "accepted_updates.jsonl")
    rejected = _read_jsonl(run_dir / "rejected_updates.jsonl")
    final_rows = [row for row in rows if row.get("split") == "test"]
    final = final_rows[-1] if final_rows else (rows[-1] if rows else {})
    failure_modes = Counter(g.get("failure_mode", "unknown") for g in gradients)
    example_gradient = gradients[0] if gradients else {}

    lines = [
        "# TextGrad-RL Experiment Summary",
        "",
        f"- Method: `{config.get('method', 'unknown')}`",
        f"- Agent: `{config.get('agent', 'unknown')}`",
        f"- Critic: `{config.get('critic', 'unknown')}`",
        f"- Tasks: train={config.get('train_tasks')}, val={config.get('val_tasks')}, test={config.get('test_tasks')}",
        f"- Platform: {environment.get('platform', 'unknown')} ({environment.get('machine', 'unknown')})",
        "",
        "## Final Metrics",
        "",
        "| split | success_rate | average_reward | test_pass_rate | invalid_action_rate | average_steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if final:
        lines.append(
            "| {split} | {success_rate} | {average_reward} | {test_pass_rate} | "
            "{invalid_action_rate} | {average_steps} |".format(
                split=final.get("split", "unknown"),
                success_rate=_fmt(final.get("success_rate")),
                average_reward=_fmt(final.get("average_reward")),
                test_pass_rate=_fmt(final.get("test_pass_rate")),
                invalid_action_rate=_fmt(final.get("invalid_action_rate")),
                average_steps=_fmt(final.get("average_steps")),
            )
        )
    lines.extend(
        [
            "",
            "## Prompt Updates",
            "",
            f"- Accepted updates: {len(accepted)}",
            f"- Rejected updates: {len(rejected)}",
            "",
            "## Common Failure Modes",
            "",
        ]
    )
    if failure_modes:
        for mode, count in failure_modes.most_common(8):
            lines.append(f"- {mode}: {count}")
    else:
        lines.append("- No textual gradients were emitted.")

    lines.extend(["", "## Example Textual Gradient", ""])
    if example_gradient:
        lines.append(f"- Target: `{example_gradient.get('target_variable_name')}`")
        lines.append(f"- Failure mode: {example_gradient.get('failure_mode')}")
        lines.append(f"- Suggested edit: {example_gradient.get('suggested_edit')}")
    else:
        lines.append("No gradient available.")

    lines.extend(["", "## Final Text Variables", ""])
    for name, variable in final_vars.items():
        value = variable.get("value", "") if isinstance(variable, dict) else ""
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```text")
        lines.append(value[:2500])
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- The default actor and critic are heuristic stand-ins for frozen small language models.",
            "- The task suite is synthetic and CPU-light, designed for rapid local iteration.",
            "- Local LLM adapters are optional and depend on a user-managed OpenAI-compatible server.",
        ]
    )
    output = run_dir / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_gradients(path: Path) -> list[dict[str, Any]]:
    gradients: list[dict[str, Any]] = []
    if not path.exists():
        return gradients
    for file in sorted(path.glob("iteration_*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        gradients.extend(item for item in payload if isinstance(item, dict))
    return gradients


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TextGrad-RL summary report.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    path = generate_summary_report(Path(args.run_dir))
    print(path)


if __name__ == "__main__":
    main()

