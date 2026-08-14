"""Tests for game-record alignment in the gameplay loop."""

from alpha_go.agents import RESIGN, Agent
from alpha_go.gameplay import play_game


class ScriptedAgent(Agent):
    """Agent that returns a fixed sequence of moves."""

    def __init__(self, moves: list[tuple[int, int]]) -> None:
        self._moves = iter(moves)

    def select_move(self, board, seed=0):  # noqa: ANN001, ARG002
        return next(self._moves)


def test_resignation_does_not_leave_an_unpaired_training_position() -> None:
    record = play_game(
        black_agent=ScriptedAgent([(0, 0)]),
        white_agent=ScriptedAgent([RESIGN]),
        board_size=9,
        max_moves=10,
        collect_boards=True,
        collect_metrics=True,
        render_debug_on_error=False,
        black_is_teacher=True,
        white_is_teacher=True,
    )

    assert record.termination == "resign"
    assert record.num_moves == 1
    assert record.moves == [(0, 0)]
    assert len(record.boards) == record.num_moves
    assert len(record.move_metrics) == record.num_moves
