from textgrad_rl.benchmarks.textarena_expanded_suites import (
    SLM_METHODS,
    SLMTextArenaRecord,
    build_slm_candidate_pool,
    normalize_textarena_action,
    slm_ppo_metrics,
    slm_update_accepted,
)
from textgrad_rl.benchmarks.action_probability import action_probability_ratio
from textgrad_rl.benchmarks.textarena_multienv_compare import canonical_env_id, num_players_for_env, rule_for_env
from textgrad_rl.benchmarks.textarena_paper_suite import initial_modular_variables
from textgrad_rl.benchmarks.textarena_policy_iteration import TextPPOConfig
from textgrad_rl.types import TextVariable


def test_variant_envs_map_to_base_policy_family() -> None:
    assert canonical_env_id("Nim-v0-large") == "Nim-v0"
    assert canonical_env_id("TowerOfHanoi-v0-hard") == "TowerOfHanoi-v0"
    assert num_players_for_env("Nim-v0-large") == 2
    assert "xor" in rule_for_env("Nim-v0-medium")


def test_normalize_textarena_action_prefers_bracketed_action() -> None:
    assert normalize_textarena_action("I will play [check].", "KuhnPoker-v0-short") == "[check]"


def test_normalize_guess_action_keeps_large_variant_range() -> None:
    assert normalize_textarena_action("I choose [50].", "GuessTheNumber-v0-hardcore") == "[50]"


def _slm_record(
    reward: float,
    success: bool,
    invalid: bool,
    actions: list[str],
    variant: str = "old",
    action_logprobs: list[float | None] | None = None,
) -> SLMTextArenaRecord:
    return SLMTextArenaRecord(
        suite="unit",
        env_id="Nim-v0",
        variant=variant,
        split="unit",
        seed=1,
        model="unit",
        target_player=0,
        reward=reward,
        success=success,
        invalid_move=invalid,
        truncated=False,
        turns=len(actions),
        reason="",
        actions=actions,
        raw_outputs=actions,
        action_logprobs=action_logprobs or [None for _ in actions],
        runtime_seconds=0.0,
    )


def test_textgrad_ppo_slm_is_registered() -> None:
    assert "textgrad_ppo_slm" in SLM_METHODS
    for method in [
        "textgrad_rl_no_gate_slm",
        "textgrad_rl_train_val_slm",
        "textgrad_rl_kl_gate_slm",
        "textgrad_rl_clipped_surrogate_slm",
        "textgrad_rl_ppo_slm",
    ]:
        assert method in SLM_METHODS


def test_slm_ppo_metrics_reward_safe_candidate() -> None:
    old = [_slm_record(-1.0, False, True, ["p0:[0]"], action_logprobs=[-3.0])]
    new = [_slm_record(1.0, True, False, ["p0:[0 3]"], variant="new", action_logprobs=[-1.0])]

    metrics = slm_ppo_metrics(old, new, old, TextPPOConfig(score_scale=5.0))

    assert metrics["surrogate_delta"] > 0.0
    assert metrics["invalid_delta"] < 0.0
    assert metrics["generated_action_logprob_delta"] == 2.0
    assert metrics["logprob_pairs"] == 1


def test_slm_update_accepts_textgrad_ppo_safe_improvement() -> None:
    variables = initial_modular_variables()
    candidate = initial_modular_variables()
    candidate["general_strategy_prompt"].value += "\n\nLearned rules:\n- Prefer a winning Nim move."
    train = [_slm_record(0.0, False, True, ["p0:[0]"])]
    train_candidate = [_slm_record(0.5, False, False, ["p0:[0 3]"], variant="new")]
    val_old = [_slm_record(0.0, False, True, ["p0:[0]"])]
    val_candidate = [_slm_record(0.5, False, False, ["p0:[0 3]"], variant="new")]

    accepted, decision = slm_update_accepted(
        "textgrad_ppo_slm",
        train,
        train_candidate,
        val_old,
        val_candidate,
        variables,
        candidate,
    )

    assert accepted
    assert decision["ppo_metrics"]["surrogate_delta"] > 0.0


def test_build_slm_candidate_pool_respects_requested_count() -> None:
    variables = {
        "general_textarena_slm_policy": TextVariable(
            name="general_textarena_slm_policy",
            value="Return one legal bracketed action.",
            role_description="General policy",
            max_chars=1200,
        ),
        "nim_slm_policy": TextVariable(
            name="nim_slm_policy",
            value="For Nim-v0: use a legal move.",
            role_description="Nim policy",
            max_chars=1200,
        ),
    }
    train = [_slm_record(0.0, False, True, ["p0:[0]"])]

    pool = build_slm_candidate_pool(
        "textgrad_rl_ppo_slm",
        "unit",
        variables,
        train,
        gradients=[],
        candidate_count=3,
    )

    assert len(pool) == 3
    assert pool[0].candidate_id == "candidate_000_textgrad"
    assert any(candidate.variables["general_textarena_slm_policy"].value != variables["general_textarena_slm_policy"].value for candidate in pool)


class _FakeActionScorer:
    def logprob(self, prompt: str, action: str) -> float:
        return -4.0 if "old" in prompt else -3.0


def test_action_probability_ratio_uses_same_action_under_both_prompts() -> None:
    ratio = action_probability_ratio(_FakeActionScorer(), "old prompt", "new prompt", "[1]")

    assert ratio.old_logprob == -4.0
    assert ratio.new_logprob == -3.0
    assert ratio.ratio > 1.0
    assert ratio.clipped_ratio == 1.2
