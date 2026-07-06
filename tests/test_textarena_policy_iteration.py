from textgrad_rl.benchmarks.textarena_multienv_compare import EpisodeRecord
from textgrad_rl.benchmarks.textarena_policy_iteration import (
    METHODS,
    TextPPOConfig,
    TextArenaValueCritic,
    advantage_assignments_from_records,
    assign_action_credits,
    ppo_gate_accepts,
    policy_ablation_gate_accepts,
    policy_objective,
    text_behavior_ratio,
    text_ppo_metrics,
)


def _record(
    env_id: str,
    reward: float,
    success: bool,
    actions: list[str],
    invalid: bool = False,
    turns: int | None = None,
) -> EpisodeRecord:
    return EpisodeRecord(
        env_id=env_id,
        variant="unit",
        split="unit",
        seed=1,
        target_side=None,
        reward=reward,
        success=success,
        done=True,
        turns=turns if turns is not None else len(actions),
        invalid_move=invalid,
        reason="",
        actions=actions,
        runtime_seconds=0.0,
    )


def test_action_credits_penalize_repeated_negative_actions() -> None:
    records = [
        _record("GuessTheNumber-v0", 1.0, True, ["p0:[10]", "p0:[15]"], turns=2),
        _record("GuessTheNumber-v0", 0.4, False, ["p0:[10]", "p0:[10]", "p0:[10]"], turns=3),
    ]
    critic = TextArenaValueCritic()
    critic.fit(records)
    credits = assign_action_credits(records, critic)

    repeated = [credit for credit in credits if "repeated_action" in credit.credit_label]
    assert repeated
    assert all(credit.advantage < 0 for credit in repeated)


def test_advantage_assignments_include_worst_action_evidence() -> None:
    records = [_record("Nim-v0", 0.0, False, ["p0:[0 1]", "p1:[0 1]"], turns=2)]
    critic = TextArenaValueCritic()
    critic.fit(records)
    credits = assign_action_credits(records, critic)
    assignments = advantage_assignments_from_records(records, credits)

    assert assignments
    assert assignments[0].target_variable_name == "nim_strategy_prompt"
    assert "worst_action" in assignments[0].evidence


def test_value_critic_estimates_environment_advantage() -> None:
    critic = TextArenaValueCritic()
    critic.fit([_record("Nim-v0", 0.0, False, ["p0:[0 1]"])])
    improved = [_record("Nim-v0", 1.0, True, ["p0:[0 3]"])]

    assert critic.estimate_advantage(improved) > 1.0


def test_policy_objective_rewards_delta_and_critic_advantage() -> None:
    details = {
        "mean_delta": 1.0,
        "new_metrics": {"average_turns": 5.0},
    }
    assignment = advantage_assignments_from_records(
        [_record("Nim-v0", 0.0, False, ["p0:[0 1]"])],
        [],
    )[0]

    assert policy_objective(details, critic_advantage=1.0, assignment=assignment) > 1.0


def test_text_behavior_ratio_moves_bad_behavior_down_on_improvement() -> None:
    assert text_behavior_ratio(old_score=0.0, new_score=1.0, advantage=-1.0, score_scale=1.0) < 1.0
    assert text_behavior_ratio(old_score=0.0, new_score=1.0, advantage=1.0, score_scale=1.0) > 1.0


def test_text_ppo_metrics_clip_surrogate_for_safe_improvement() -> None:
    old_records = [
        _record("Nim-v0", 0.0, False, ["p0:[0 1]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    new_records = [
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    critic = TextArenaValueCritic()
    critic.fit(old_records)

    metrics = text_ppo_metrics(old_records, new_records, critic, TextPPOConfig(clip_epsilon=0.2))

    assert metrics["surrogate_delta"] > 0.0
    assert metrics["clip_fraction"] > 0.0
    assert metrics["invalid_delta"] == 0.0


def test_text_ppo_metrics_use_fallback_action_advantage_for_flat_baseline() -> None:
    old_records = [
        _record("TowerOfHanoi-v0", 0.0, False, ["p0:[A B]"], turns=15),
        _record("TowerOfHanoi-v0", 0.0, False, ["p0:[A B]"], turns=15),
    ]
    new_records = [
        _record("TowerOfHanoi-v0", 1.0, True, ["p0:[A C]"], turns=7),
        _record("TowerOfHanoi-v0", 1.0, True, ["p0:[A C]"], turns=7),
    ]
    critic = TextArenaValueCritic()
    critic.fit(old_records)

    metrics = text_ppo_metrics(
        old_records,
        new_records,
        critic,
        TextPPOConfig(score_scale=5.0),
        fallback_advantage=-1.0,
    )

    assert metrics["surrogate_delta"] > 0.0
    assert metrics["fallback_advantage"] == -1.0


def test_ppo_gate_accepts_trust_region_safe_candidate() -> None:
    old_records = [
        _record("Nim-v0", 0.0, False, ["p0:[0 1]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    new_records = [
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    critic = TextArenaValueCritic()
    critic.fit(old_records)

    accepted, details = ppo_gate_accepts(
        old_records,
        new_records,
        critic,
        TextPPOConfig(clip_epsilon=0.2, target_kl=1.0, score_scale=2.0, min_surrogate_delta=0.001),
        seed=123,
        min_mean_delta=0.001,
        max_ci_low_regression=0.0,
    )

    assert accepted
    assert details["bootstrap_accepted"]
    assert details["ppo_objective"] > 0.0


def test_policy_ablation_methods_are_registered() -> None:
    for method in [
        "textgrad_rl_no_gate",
        "textgrad_rl_train_val",
        "textgrad_rl_kl_gate",
        "textgrad_rl_clipped_surrogate",
        "textgrad_rl_ppo",
    ]:
        assert method in METHODS


def test_policy_ablation_gates_separate_kl_and_surrogate() -> None:
    old_records = [
        _record("Nim-v0", 0.0, False, ["p0:[0 1]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    new_records = [
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
        _record("Nim-v0", 1.0, True, ["p0:[0 3]"], turns=1),
    ]
    critic = TextArenaValueCritic()
    critic.fit(old_records)
    config = TextPPOConfig(clip_epsilon=0.2, target_kl=1.0, score_scale=2.0, min_surrogate_delta=0.001)

    kl_accepted, kl_details = policy_ablation_gate_accepts(
        "kl_gate",
        old_records,
        new_records,
        critic,
        config,
        seed=123,
        min_mean_delta=0.001,
        max_ci_low_regression=0.0,
    )
    clip_accepted, clip_details = policy_ablation_gate_accepts(
        "clipped_surrogate",
        old_records,
        new_records,
        critic,
        config,
        seed=123,
        min_mean_delta=0.001,
        max_ci_low_regression=0.0,
    )

    assert kl_accepted
    assert kl_details["gate_mode"] == "kl_gate"
    assert clip_accepted
    assert clip_details["gate_mode"] == "clipped_surrogate"
