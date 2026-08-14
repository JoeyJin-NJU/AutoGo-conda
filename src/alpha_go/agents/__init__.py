"""Agent implementations for Go."""
from alpha_go.agents.base import Agent, get_agent, list_agents, register_agent, PASS, RESIGN
from alpha_go.agents.nn_agent import NNAgent, BatchedNNAgent
from alpha_go.agents.nn_mcts import NNMCTSAgent, NNEvaluator
from alpha_go.go import FastGoBoard, GoState, BLACK, WHITE, EMPTY

# Import agents to trigger registration
from alpha_go.agents import random as _random  # noqa: F401
from alpha_go.agents import nn_agent as _nn_agent  # noqa: F401

__all__ = [
    "Agent",
    "PASS",
    "get_agent",
    "list_agents",
    "register_agent",
    "NNAgent",
    "BatchedNNAgent",
    "NNMCTSAgent",
    "NNEvaluator",
    "FastGoBoard",
    "GoState",
    "BLACK",
    "WHITE",
    "EMPTY",
]
