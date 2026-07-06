from textgrad_rl.critics.heuristic_critic import HeuristicTextualCritic
from textgrad_rl.text_variables import initial_text_variables
from textgrad_rl.types import Action, StepRecord, Trajectory


def test_critic_emits_shape_and_forbidden_gradients():
    trajectory = Trajectory(
        task_id="task",
        task_family="shape_mismatch_training_crash",
        steps=[
            StepRecord(
                step_index=0,
                observation_summary="",
                action=Action(type="edit_file", path="tests/test_training.py", content="assert True"),
                reward=-8.0,
                done=False,
                info={"invalid_reason": "editing tests is forbidden"},
                stdout="",
                stderr="editing tests is forbidden",
                files_changed=[],
            ),
            StepRecord(
                step_index=1,
                observation_summary="",
                action=Action(type="run_training", command="python train.py"),
                reward=-1.0,
                done=True,
                info={},
                stdout="",
                stderr="ValueError: matmul dimension mismatch",
                files_changed=[],
            ),
        ],
        total_reward=-9.0,
        success=False,
        final_status="failed",
        failure_summary="matmul dimension mismatch",
        file_diffs={},
        test_results={"passed": False},
        metric_results={},
        command_count=1,
        invalid_action_count=1,
        runtime_seconds=0.1,
    )
    gradients = HeuristicTextualCritic().critique(trajectory, initial_text_variables())
    modes = {gradient.failure_mode for gradient in gradients}
    assert "Invalid or forbidden edit attempted" in modes
    assert "Shape mismatch not converted into targeted inspection" in modes

