from textgrad_rl.optim.acceptance_gate import AcceptanceGate
from textgrad_rl.text_variables import initial_text_variables
from textgrad_rl.envs.task_factory import create_task_spec


def test_acceptance_gate_evaluates_validation_task():
    variables = initial_text_variables()
    task = create_task_spec("shape_mismatch_training_crash", 11, split="val")
    gate = AcceptanceGate(max_steps=15)
    metrics = gate.evaluate(variables, [task])
    assert metrics.split == "val"
    assert metrics.average_steps > 0

