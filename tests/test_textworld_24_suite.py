from textgrad_rl.benchmarks.textworld_24_suite import (
    METHODS,
    TextWorldRecord,
    auxiliary_specs,
    default_specs,
    learned_rule_ids,
    match_wanted_action,
    parse_objective_actions,
    parse_room,
    recipe_line_matches_action,
    variables_from_diagnostics,
)


def test_parse_objective_actions_handles_textworld_phrases() -> None:
    objective = (
        "First off, make an effort to take a trip south. Then, pick-up the fondue from the floor "
        "of the bedchamber. Check that the wooden door is unlocked with the old key. "
        "After that, place the bell pepper on the stove."
    )
    actions = parse_objective_actions(objective)

    assert "go south" in actions
    assert "take fondue" in actions
    assert "unlock wooden door with old key" in actions
    assert "put bell pepper on stove" in actions


def test_parse_objective_actions_preserves_repeated_moves_and_synonyms() -> None:
    objective = (
        "First, go west. Then, make an attempt to take a trip south. "
        "Then, attempt to go to the south. Then, recover the coin from the floor. "
        "Ensure that the screen door inside the kitchen is open."
    )
    assert parse_objective_actions(objective) == [
        "go west",
        "go south",
        "go south",
        "take coin",
        "open screen door",
    ]


def test_match_wanted_action_handles_take_from_container() -> None:
    assert (
        match_wanted_action("take old key", ["take old key from antique trunk", "look"])
        == "take old key from antique trunk"
    )


def test_match_wanted_action_handles_unlock_with_inferred_key() -> None:
    assert (
        match_wanted_action("unlock wooden door", ["unlock wooden door with old key"])
        == "unlock wooden door with old key"
    )


def test_match_wanted_action_handles_put_destination() -> None:
    assert match_wanted_action("put bell pepper on stove", ["put bell pepper on stove"]) == "put bell pepper on stove"


def test_parse_room_from_textworld_description() -> None:
    assert parse_room("-= Kitchen =-\nYou are in a kitchen.") == "kitchen"


def test_recipe_line_matches_cooking_actions() -> None:
    assert recipe_line_matches_action("slice yellow potato", "slice yellow potato")
    assert recipe_line_matches_action("roast banana", "cook banana in oven")


def test_retry_with_diagnostics_is_registered() -> None:
    assert "retry_with_diagnostics" in METHODS
    assert "ungated_persistent_rules" in METHODS


def test_optimization_specs_are_disjoint_from_test_specs() -> None:
    test = default_specs(62001)
    train = auxiliary_specs(72001, "train")
    validation = auxiliary_specs(82001, "validation")

    assert len(test) == 24
    assert len(train) == len(validation) == 8
    assert {spec.spec_id for spec in test}.isdisjoint(spec.spec_id for spec in train)
    assert {spec.spec_id for spec in test}.isdisjoint(spec.spec_id for spec in validation)
    assert {spec.seed for spec in test}.isdisjoint(spec.seed for spec in train)
    assert {spec.seed for spec in test}.isdisjoint(spec.seed for spec in validation)


def test_failed_trajectory_produces_task_local_diagnostic_rules() -> None:
    record = TextWorldRecord(
        benchmark="textworld_24",
        spec_id="treasure_level_15",
        family="tw-treasure_hunter",
        category="navigation_objective",
        method="retry_with_diagnostics",
        split="test_attempt_1",
        seed=62204,
        success=False,
        reward=0.0,
        invalid_action=False,
        repeated_actions=True,
        turns=80,
        final_score=0.0,
        max_score=1.0,
        actions=["go east", "go north"] * 40,
        failure_reason="step budget exhausted",
        runtime_seconds=0.1,
        total_turns=80,
    )

    variables, gradients = variables_from_diagnostics([record])

    assert {gradient.failure_mode for gradient in gradients} == {
        "textworld_24:graph_exploration",
        "textworld_24:objective_sequence",
    }
    assert learned_rule_ids(variables) == {"graph_exploration", "objective_sequence"}
