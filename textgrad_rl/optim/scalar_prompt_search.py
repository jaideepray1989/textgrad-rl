"""Scalar-only prompt mutation baseline."""

from __future__ import annotations

import copy
import random

from textgrad_rl.types import TextVariable


MUTATIONS = [
    ("experiment_planning_prompt", "When a command fails, inspect logs before repeating the same command."),
    ("patch_planning_prompt", "Prefer minimal source-code edits grounded in the current file content."),
    ("validation_prompt", "Before submit, run tests, training, and eval when available."),
    ("triage_prompt", "Classify whether the failure is schema, shape, metric, latency, or reproducibility related."),
]


class ScalarPromptSearch:
    """Generate simple prompt mutations without textual trajectory gradients."""

    def propose(
        self,
        text_variables: dict[str, TextVariable],
        seed: int,
        iteration: int,
    ) -> dict[str, TextVariable]:
        rng = random.Random(seed + iteration * 101)
        updated = copy.deepcopy(text_variables)
        name, rule = rng.choice(MUTATIONS)
        if name not in updated:
            name = next(iter(updated))
        variable = updated[name]
        if rule.lower() not in variable.value.lower():
            if "Learned rules:" not in variable.value:
                variable.value = variable.value.rstrip() + "\n\nLearned rules:"
            variable.value += f"\n- {rule}"
            variable.version += 1
            variable.gradient_history.append(f"scalar_prompt_search: {rule}")
        return updated

