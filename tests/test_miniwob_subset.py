from textgrad_rl.benchmarks.miniwob_subset import (
    PromptAwareMiniWobAgent,
    extract_elements,
    initial_miniwob_variables,
    prompt_has_submit_after_select_rule,
    select_targets,
)


def _obs(goal: str, nodes: list[dict], props: dict[str, dict] | None = None) -> dict:
    return {
        "goal": goal,
        "axtree_object": {"nodes": nodes},
        "extra_element_properties": props or {node["browsergym_id"]: {"clickable": True} for node in nodes},
    }


def _node(bid: str, role: str, name: str) -> dict:
    return {
        "browsergym_id": bid,
        "role": {"value": role},
        "name": {"value": name},
    }


def test_select_targets_splits_checkbox_goal() -> None:
    assert select_targets("Select Jc, s6WcI, XMHY and click Submit.") == ["Jc", "s6WcI", "XMHY"]


def test_fixed_actor_repeats_selection_without_submit_rule() -> None:
    variables = initial_miniwob_variables()
    agent = PromptAwareMiniWobAgent(variables)
    obs = _obs(
        "Select Nb and click Submit.",
        [_node("21", "checkbox", "Nb"), _node("27", "button", "Submit")],
    )

    first = agent.act(obs, [])
    second = agent.act(obs, [first])

    assert first == 'click("21")'
    assert second == first


def test_textgrad_rule_clicks_submit_after_selection() -> None:
    variables = initial_miniwob_variables()
    variables["general_agent_policy"].value += (
        "\n- After selecting all required checkbox, radio, or list options, click Submit exactly once."
    )
    agent = PromptAwareMiniWobAgent(variables)
    obs = _obs(
        "Select QWS1FL and click Submit.",
        [_node("24", "radio", "QWS1FL"), _node("27", "button", "Submit")],
    )

    assert prompt_has_submit_after_select_rule(variables)
    assert agent.act(obs, ['click("24")']) == 'click("27")'


def test_extract_elements_keeps_clickable_and_form_controls() -> None:
    obs = _obs(
        "Focus into the textbox.",
        [_node("12", "textbox", ""), _node("99", "generic", "")],
        {"12": {"clickable": False}, "99": {"clickable": False}},
    )

    elements = extract_elements(obs)

    assert [element.bid for element in elements] == ["12"]
