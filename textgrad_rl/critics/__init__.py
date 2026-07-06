"""Trajectory critics."""

from textgrad_rl.critics.heuristic_critic import HeuristicTextualCritic
from textgrad_rl.critics.llm_critic import LLMTrajectoryCritic

__all__ = ["HeuristicTextualCritic", "LLMTrajectoryCritic"]

