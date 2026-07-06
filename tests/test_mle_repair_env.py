from pathlib import Path

from textgrad_rl.envs.mle_repair_env import MLERepairEnv
from textgrad_rl.envs.task_factory import create_task_spec
from textgrad_rl.types import Action


def test_shape_task_can_be_repaired_and_submitted(tmp_path: Path):
    spec = create_task_spec("shape_mismatch_training_crash", 3)
    env = MLERepairEnv(spec, tmp_path, max_steps=10)
    env.reset()
    env.step(Action(type="run_tests"))
    repaired = spec.files["train.py"].replace("weights = np.ones(3)", "weights = np.ones(X.shape[1])")
    env.step(Action(type="edit_file", path="train.py", content=repaired))
    env.step(Action(type="run_tests"))
    env.step(Action(type="run_training"))
    env.step(Action(type="run_eval"))
    _, _, done, info = env.step(Action(type="submit_patch"))
    trajectory = env.get_trajectory()
    assert done
    assert info["hidden_validation_passed"]
    assert trajectory.success
    assert "train.py" in trajectory.file_diffs

