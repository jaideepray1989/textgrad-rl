from textgrad_rl.benchmarks.textarena_broad_suite import (
    BROAD_TEXTARENA_ENVS,
    METHODS,
    diagnostic_variables,
    parse_bracketed_actions,
    support_group,
)


def test_broad_suite_has_50_unique_envs() -> None:
    env_ids = [spec.env_id for spec in BROAD_TEXTARENA_ENVS]

    assert len(env_ids) == 50
    assert len(set(env_ids)) == 50


def test_broad_suite_marks_supported_policy_variants() -> None:
    assert support_group("Nim-v0-medium") == "supported_policy_family"
    assert support_group("TowerOfHanoi-v0-medium") == "supported_policy_family"
    assert support_group("Wordle-v0") == "generic_fallback"


def test_parse_bracketed_actions_deduplicates_legal_actions() -> None:
    observation = "Available Moves: '[0]', '[1]', '[1]'. Valid moves: [Up], [Down]."

    assert parse_bracketed_actions(observation) == ["[0]", "[1]", "[Up]", "[Down]"]


def test_rulepi_control_methods_are_registered() -> None:
    assert "retry_with_diagnostics" in METHODS
    assert "ungated_persistent_rules" in METHODS


def test_retry_diagnostic_adds_only_the_failed_environment_rule() -> None:
    variables, gradients = diagnostic_variables("Nim-v0")

    assert len(gradients) == 1
    assert gradients[0].target_variable_name == "nim_strategy_prompt"
    assert variables["nim_strategy_prompt"].version == 1
    assert variables["connectfour_strategy_prompt"].version == 0
