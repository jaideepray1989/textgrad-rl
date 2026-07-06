"""Ablation method names and descriptions."""

ABLATIONS = {
    "fixed_prompt": "No prompt updates; evaluate the initial text variables.",
    "scalar_prompt_search": "Prompt mutations based only on scalar outcomes.",
    "modular_textgrad": "Trajectory-level textual gradients targeted to modular prompts.",
    "monolithic_textgrad": "Textual gradients applied to one concatenated prompt.",
    "no_acceptance_gate": "Apply textual updates without validation acceptance.",
}

