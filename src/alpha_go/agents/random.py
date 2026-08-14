"""Random agent implementation."""
from __future__ import annotations

import random

import alpha_go_cpp

from alpha_go.agents.base import Agent, PASS, register_agent


@register_agent("random")
class RandomAgent(Agent):
    """Agent that plays random legal moves."""

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        legal_flat = board.get_legal_moves_flat()
        if not legal_flat:
            return PASS
        random.seed(seed)
        flat_idx = random.choice(legal_flat)
        row, col = board.row_col(flat_idx)
        return (row, col)
