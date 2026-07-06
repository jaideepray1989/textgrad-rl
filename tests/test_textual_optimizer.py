from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.text_variables import initial_text_variables
from textgrad_rl.types import TextualGradient


def test_textual_optimizer_updates_target_prompt_once():
    variables = initial_text_variables()
    gradient = TextualGradient(
        target_variable_name="experiment_planning_prompt",
        failure_mode="Repeated full training",
        evidence_from_trajectory="train.py failed twice",
        gradient_text="Inspect logs before rerunning.",
        suggested_edit="Add a rule: after deterministic crashes, inspect logs before rerunning full training.",
        confidence=0.9,
        forbidden_shortcuts=[],
    )
    updated = TextualGradientDescent().step(variables, [gradient, gradient], [])
    value = updated["experiment_planning_prompt"].value
    assert updated["experiment_planning_prompt"].version == 1
    assert value.count("after deterministic crashes") == 1
    assert variables["experiment_planning_prompt"].version == 0

