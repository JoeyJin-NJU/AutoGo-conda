"""Base agent class and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

import alpha_go_cpp

# Pass action represented as special coordinate (-1, -1)
PASS: tuple[int, int] = (-1, -1)
# Resign action represented as special coordinate (-2, -2)
RESIGN: tuple[int, int] = (-2, -2)


def get_pass_index(board_size: int) -> int:
    """Get flat index for pass action: board_size^2."""
    return board_size * board_size

_AGENT_REGISTRY: dict[str, type[Agent]] = {}


class Agent(ABC):
    """Base class for Go agents."""

    def start_game(self, board_size: int) -> None:
        """Called at start of game. Override to initialize internal state."""

    def notify_move(self, row: int, col: int) -> None:
        """Called when any move is played (use PASS=(-1,-1) for pass). Override to sync internal state."""

    def end_game(self) -> None:
        """Called at end of game. Override to cleanup resources."""

    @property
    def checkpoint_path(self) -> str | None:
        """Return checkpoint path if agent uses one, None otherwise."""
        return None

    @abstractmethod
    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select a move given current game state. Returns (row, col) or PASS."""


def register_agent(name: str):
    """Decorator to register an agent class."""
    def decorator(cls: type[Agent]) -> type[Agent]:
        _AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def get_agent(name: str) -> Agent:
    """Get an instance of a registered agent by name."""
    if name not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return _AGENT_REGISTRY[name]()


def list_agents() -> list[str]:
    """List all registered agent names."""
    return list(_AGENT_REGISTRY.keys())
