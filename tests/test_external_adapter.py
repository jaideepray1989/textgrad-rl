from textgrad_rl.benchmarks.external_adapter import (
    ExternalAgentEpisode,
    ExternalAgentStep,
    external_episode_score,
    gradients_from_external_episodes,
)


def test_external_episode_score_penalizes_invalid_and_truncated_runs() -> None:
    good = ExternalAgentEpisode(
        benchmark="swebench",
        task_id="repo__issue-1",
        split="test",
        seed=1,
        success=True,
        reward=1.0,
        invalid_action=False,
        truncated=False,
        steps=[ExternalAgentStep("obs", "edit")],
    )
    bad = ExternalAgentEpisode(
        benchmark="swebench",
        task_id="repo__issue-2",
        split="test",
        seed=2,
        success=False,
        reward=0.0,
        invalid_action=True,
        truncated=True,
        steps=[ExternalAgentStep("obs", "edit"), ExternalAgentStep("obs2", "edit")],
    )

    assert external_episode_score(good) > external_episode_score(bad)


def test_gradients_from_external_episodes_targets_policy_variable() -> None:
    episode = ExternalAgentEpisode(
        benchmark="webarena",
        task_id="shopping-1",
        split="train",
        seed=7,
        success=False,
        reward=0.0,
        invalid_action=True,
        truncated=False,
        steps=[ExternalAgentStep("obs", "click button"), ExternalAgentStep("obs", "click button")],
        target_variable="browser_agent_policy",
        failure_reason="repeated invalid click",
    )

    gradients = gradients_from_external_episodes([episode])

    assert len(gradients) == 1
    assert gradients[0].target_variable_name == "browser_agent_policy"
    assert "repeated invalid click" in gradients[0].evidence_from_trajectory
