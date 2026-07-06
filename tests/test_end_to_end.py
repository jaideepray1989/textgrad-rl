import json
from pathlib import Path

from textgrad_rl.experiments.runner import run_experiment
from textgrad_rl.utils.json_utils import read_json


def test_one_iteration_end_to_end_run(tmp_path: Path):
    output_dir = tmp_path / "run"
    run_experiment(
        {
            "agent": "heuristic",
            "critic": "heuristic",
            "method": "no_acceptance_gate",
            "iterations": 1,
            "train_tasks": 2,
            "val_tasks": 1,
            "test_tasks": 1,
            "max_steps": 15,
            "seed": 17,
            "output_dir": str(output_dir),
            "command_timeout_sec": 10,
            "local_llm_base_url": None,
            "local_llm_model": None,
        }
    )
    metrics_path = output_dir / "metrics.csv"
    final_vars_path = output_dir / "final_text_variables.json"
    gradients_path = output_dir / "gradients" / "iteration_001.json"
    assert metrics_path.exists()
    assert final_vars_path.exists()
    gradients = json.loads(gradients_path.read_text(encoding="utf-8"))
    assert gradients
    final_vars = read_json(final_vars_path)
    assert any(variable["version"] > 0 for variable in final_vars.values())
    trajectory_files = list((output_dir / "trajectories").rglob("*.json"))
    assert trajectory_files
    for path in trajectory_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not any(diff_path.startswith("tests/") for diff_path in payload.get("file_diffs", {}))

