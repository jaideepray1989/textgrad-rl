from textgrad_rl.benchmarks.textworld_24_suite import (
    match_wanted_action,
    parse_objective_actions,
    parse_room,
    recipe_line_matches_action,
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
