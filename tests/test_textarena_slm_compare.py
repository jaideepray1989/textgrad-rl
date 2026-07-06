from textgrad_rl.benchmarks.textarena_slm_compare import normalize_guess_action


def test_normalize_guess_action_prefers_bracketed_value() -> None:
    assert normalize_guess_action("I choose [7] because it is central.") == "[7]"


def test_normalize_guess_action_coerces_plain_number() -> None:
    assert normalize_guess_action("The best next guess is 13.") == "[13]"


def test_normalize_guess_action_clamps_out_of_range_number() -> None:
    assert normalize_guess_action("I choose [21].") == "[20]"


def test_normalize_guess_action_falls_back_to_midpoint() -> None:
    assert normalize_guess_action("I cannot decide.") == "[10]"
