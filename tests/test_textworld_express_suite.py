from textgrad_rl.benchmarks.textworld_express_suite import (
    initial_textworld_variables,
    learned_rule_ids,
    parse_arithmetic_answer,
    ppo_gate_stats,
    score_twc_put_action,
    simonsays_action,
    sorting_action,
    TextWorldExpressState,
)
from textgrad_rl.types import TextVariable


def test_simonsays_action_uses_quoted_command() -> None:
    assert simonsays_action("Simon says, 'slice grape'.", ["touch orange", "slice grape"]) == "slice grape"


def test_sorting_action_takes_smallest_quantity() -> None:
    state = TextWorldExpressState(game="sorting")
    assert sorting_action(state, ["take 35 strawberries", "take 4 grapes", "take 26 apples"]) == "take 4 grapes"
    state.previous_actions.extend(["take 4 grapes", "put 4 grapes in box"])
    assert sorting_action(state, ["take 35 strawberries", "take 4 grapes", "take 26 apples"]) == "take 26 apples"
    state = TextWorldExpressState(game="sorting")
    state.previous_actions.extend(["take 3ml of brass", "put 3ml of brass in box"])
    assert sorting_action(state, ["take 6l of aluminum", "take 18ml of aluminum"]) == "take 18ml of aluminum"


def test_arithmetic_parser_handles_text_problem() -> None:
    assert parse_arithmetic_answer("divide 36 by 6") == 6
    assert parse_arithmetic_answer("subtract 3 from 10") == 7
    assert parse_arithmetic_answer("multiply 8 by 9") == 72


def test_twc_storage_scores_semantic_targets() -> None:
    assert score_twc_put_action("put blue golf shoes in shoe cabinet") > score_twc_put_action("put blue golf shoes in hat rack")
    assert score_twc_put_action("put beret in hat rack") > score_twc_put_action("put beret in shoe cabinet")


def test_learned_rule_ids_detects_optimizer_rules() -> None:
    variables = initial_textworld_variables()
    variables["textworld_policy"] = TextVariable(
        name="textworld_policy",
        value=variables["textworld_policy"].value
        + "\n- For simonsays, parse the quoted command after 'Simon says' and execute exactly that visible action.",
        role_description="policy",
    )
    assert "simonsays" in learned_rule_ids(variables)


def test_ppo_gate_stats_reports_action_distance() -> None:
    from textgrad_rl.benchmarks.textworld_express_suite import TextWorldExpressRecord

    old = TextWorldExpressRecord(
        benchmark="textworld_express",
        game="simonsays",
        category="instruction_following",
        method="old",
        split="dev",
        seed=1,
        success=False,
        reward=0.0,
        invalid_action=False,
        repeated_actions=True,
        turns=2,
        final_score=0.0,
        actions=["touch orange", "touch orange"],
        failure_reason="repeat",
        runtime_seconds=0.0,
    )
    new = TextWorldExpressRecord(
        benchmark="textworld_express",
        game="simonsays",
        category="instruction_following",
        method="new",
        split="dev",
        seed=1,
        success=True,
        reward=1.0,
        invalid_action=False,
        repeated_actions=False,
        turns=1,
        final_score=1.0,
        actions=["slice grape"],
        failure_reason="",
        runtime_seconds=0.0,
    )
    stats = ppo_gate_stats([old], [new], clip_epsilon=0.2)
    assert stats["approx_kl"] > 0
    assert stats["clipped_surrogate_delta"] > 0
