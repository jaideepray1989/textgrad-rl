from pathlib import Path

from textgrad_rl.agents.prompt_aware_heuristic_agent import PromptAwareHeuristicAgent
from textgrad_rl.envs.mle_repair_env import MLERepairEnv
from textgrad_rl.envs.task_factory import create_task_spec
from textgrad_rl.text_variables import initial_text_variables
from textgrad_rl.types import Action


def test_heuristic_agent_solves_shape_task(tmp_path: Path):
    spec = create_task_spec("shape_mismatch_training_crash", 5)
    env = MLERepairEnv(spec, tmp_path, max_steps=15)
    agent = PromptAwareHeuristicAgent()
    variables = initial_text_variables()
    observation = env.reset()
    done = False
    while not done and observation.remaining_steps > 0:
        action = agent.act(observation, variables)
        observation, _, done, _ = env.step(action)
    if not done:
        env.step(Action(type="submit_patch"))
    trajectory = env.get_trajectory()
    assert trajectory.success
    assert any("train.py" in step.files_changed for step in trajectory.steps)

