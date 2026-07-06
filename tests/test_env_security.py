from pathlib import Path

from textgrad_rl.envs.mle_repair_env import MLERepairEnv
from textgrad_rl.envs.task_factory import create_task_spec
from textgrad_rl.types import Action


def test_rejects_test_edits(tmp_path: Path):
    env = MLERepairEnv(create_task_spec("shape_mismatch_training_crash", 1), tmp_path)
    env.reset()
    _, reward, _, info = env.step(
        Action(type="edit_file", path="tests/test_training.py", content="def test_x():\n    assert True\n")
    )
    assert reward < -2
    assert info["invalid"]
    assert "forbidden" in info["invalid_reason"] or "test" in info["invalid_reason"]


def test_rejects_path_traversal(tmp_path: Path):
    env = MLERepairEnv(create_task_spec("shape_mismatch_training_crash", 1), tmp_path)
    env.reset()
    _, _, _, info = env.step(Action(type="edit_file", path="../escape.py", content="print('x')\n"))
    assert info["invalid"]
    assert "outside" in info["invalid_reason"]


def test_rejects_hidden_reads(tmp_path: Path):
    env = MLERepairEnv(create_task_spec("shape_mismatch_training_crash", 1), tmp_path)
    env.reset()
    _, _, _, info = env.step(Action(type="read_file", path="hidden_validation.py"))
    assert info["invalid"]
    assert "hidden" in info["invalid_reason"]

