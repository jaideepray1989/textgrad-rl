from textgrad_rl.benchmarks.miniwob_subset import (
    FIFTY_TASK_ENVS,
    MiniWobElement,
    PromptAwareMiniWobAgent,
    category_for_env,
    extract_elements,
    initial_miniwob_variables,
    normalize_llm_action,
    prompt_has_submit_after_select_rule,
    resolve_env_suite,
    select_targets,
    text_variables_changed,
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


def test_login_goal_fills_password_before_clicking_login() -> None:
    agent = PromptAwareMiniWobAgent(initial_miniwob_variables())
    obs = _obs(
        'Enter the username "cierra" and the password "11L" into the text fields and press login.',
        [_node("20", "button", "Login"), _node("16", "textbox", ""), _node("19", "textbox", "")],
    )

    assert agent.act(obs, ['fill("16", "cierra")']) == 'fill("19", "11L")'


def test_resolve_50_task_suite_and_categories() -> None:
    envs = resolve_env_suite("50")

    assert len(envs) == 50
    assert envs == FIFTY_TASK_ENVS
    assert category_for_env("email-inbox") == "simulated_app"


def test_normalize_llm_action_repairs_verbose_bid() -> None:
    elements = [MiniWobElement(bid="13", role="button", name="Click Me", clickable=True)]

    assert normalize_llm_action('click("13 button Click Me.")', elements) == 'click("13")'


def test_normalize_llm_action_extracts_fill_call() -> None:
    elements = [MiniWobElement(bid="14", role="textbox", name="", clickable=True)]

    assert normalize_llm_action('Action: fill("14", "Myron")', elements) == 'fill("14", "Myron")'


def test_text_variables_changed_requires_real_prompt_delta() -> None:
    old = initial_miniwob_variables()
    same = initial_miniwob_variables()
    changed = initial_miniwob_variables()
    changed["general_agent_policy"].value += "\n- New rule."

    assert not text_variables_changed(old, same)
    assert text_variables_changed(old, changed)
