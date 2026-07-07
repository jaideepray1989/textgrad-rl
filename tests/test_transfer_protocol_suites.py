from pathlib import Path

from textgrad_rl.benchmarks.transfer_protocol_suites import (
    TASKS_BY_SUITE,
    evaluate_records,
    gradients_from_transfer_records,
    initial_variables,
    parse_suites,
    policy_rule_ids,
    run_transfer_suite,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent


def test_parse_suites_rejects_unknown_suite() -> None:
    try:
        parse_suites("browser_transfer,nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("Expected an unknown-suite error")


def test_browser_transfer_training_learns_selection_rule(tmp_path: Path) -> None:
    suite = "browser_transfer"
    variables = initial_variables(suite)
    train = [task for task in TASKS_BY_SUITE[suite] if task.split == "train"]
    records, _episodes = evaluate_records(train, "fixed_policy", variables, 1, tmp_path / "train.jsonl")

    gradients = gradients_from_transfer_records(records)
    updated = TextualGradientDescent(max_rules_per_step=12).step(
        variables,
        gradients,
        constraints=["must not inspect hidden labels", "must not mutate benchmark data"],
    )

    assert "select_then_submit" not in policy_rule_ids(variables)
    assert "select_then_submit" in policy_rule_ids(updated)
    assert "avoid_repeated_browser_action" in policy_rule_ids(updated)


def test_transfer_suite_improves_target_success(tmp_path: Path) -> None:
    result = run_transfer_suite("tau_transfer", tmp_path, seed=7)
    rows = {row["method"]: row for row in result["summary"]}

    assert result["accepted"]
    assert rows["textgrad_rl"]["success_rate"] > rows["fixed_policy"]["success_rate"]
