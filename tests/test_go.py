"""Tests for GNU Go GTP integration."""
import subprocess

import pytest

from alpha_go import BLACK, WHITE, GTPEngine


def gnugo_available() -> bool:
    """Check if GNU Go is installed."""
    try:
        result = subprocess.run(
            ["gnugo", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not gnugo_available(), reason="GNU Go not installed")
class TestGTPEngine:
    """Tests for GTPEngine class."""

    def test_new_game_9x9(self) -> None:
        """Create a new 9x9 game."""
        engine = GTPEngine.new(size=9)
        try:
            assert engine.size == 9
            assert engine.to_play == BLACK
            assert not engine.is_over()

            board = engine.get_board()
            assert len(board) == 9
            assert len(board[0]) == 9
            assert all(cell == 0 for row in board for cell in row)
        finally:
            engine.close()

    def test_new_game_19x19(self) -> None:
        """Create a new 19x19 game."""
        engine = GTPEngine.new(size=19)
        try:
            assert engine.size == 19
            board = engine.get_board()
            assert len(board) == 19
        finally:
            engine.close()

    def test_play_move(self) -> None:
        """Play a move and verify board state."""
        engine = GTPEngine.new(size=9)
        try:
            # Black plays at D5 (row=4, col=3)
            success = engine.play(4, 3)
            assert success
            assert engine.to_play == WHITE
            assert engine.last_move == (4, 3)

            board = engine.get_board()
            assert board[4][3] == BLACK
        finally:
            engine.close()

    def test_alternating_turns(self) -> None:
        """Players alternate turns."""
        engine = GTPEngine.new(size=9)
        try:
            engine.play(0, 0)  # Black
            assert engine.to_play == WHITE

            engine.play(1, 1)  # White
            assert engine.to_play == BLACK

            board = engine.get_board()
            assert board[0][0] == BLACK
            assert board[1][1] == WHITE
        finally:
            engine.close()

    def test_illegal_move_occupied(self) -> None:
        """Cannot play on occupied intersection."""
        engine = GTPEngine.new(size=9)
        try:
            engine.play(4, 4)
            assert not engine.is_legal(4, 4)
        finally:
            engine.close()

    def test_pass_move(self) -> None:
        """Pass move switches turn."""
        engine = GTPEngine.new(size=9)
        try:
            engine.play(None, None)  # Black passes
            assert engine.to_play == WHITE
            assert engine.last_move is None
            assert engine.consecutive_passes == 1
        finally:
            engine.close()

    def test_game_ends_after_two_passes(self) -> None:
        """Game ends after two consecutive passes."""
        engine = GTPEngine.new(size=9)
        try:
            engine.play(None, None)  # Black passes
            assert not engine.is_over()

            engine.play(None, None)  # White passes
            assert engine.is_over()
            assert engine.result() is not None
        finally:
            engine.close()

    def test_genmove(self) -> None:
        """GNU Go can generate moves."""
        engine = GTPEngine.new(size=9, level=1)
        try:
            move = engine.genmove()
            # GNU Go should make a move (not pass) on empty board
            assert move is not None
            assert engine.to_play == WHITE

            board = engine.get_board()
            assert board[move[0]][move[1]] == BLACK
        finally:
            engine.close()

    def test_legal_moves(self) -> None:
        """Get list of legal moves."""
        engine = GTPEngine.new(size=9)
        try:
            moves = engine.get_legal_moves()
            # All 81 positions should be legal on empty board
            assert len(moves) == 81

            # Play a move
            engine.play(4, 4)
            moves = engine.get_legal_moves()
            # Now 80 positions should be legal
            assert len(moves) == 80
            assert (4, 4) not in moves
        finally:
            engine.close()

    def test_capture(self) -> None:
        """Stones are captured when surrounded."""
        engine = GTPEngine.new(size=9)
        try:
            # Set up a capture scenario in corner
            # White at A9 (0,0), Black surrounds at B9 (0,1) and A8 (1,0)
            engine.play(0, 1)  # Black at B9
            engine.play(0, 0)  # White at A9
            engine.play(1, 0)  # Black at A8 - captures white

            board = engine.get_board()
            assert board[0][0] == 0  # White stone captured
            assert board[0][1] == BLACK
            assert board[1][0] == BLACK
        finally:
            engine.close()

    def test_coord_conversion(self) -> None:
        """Coordinate conversion is correct."""
        engine = GTPEngine.new(size=9)
        try:
            # Test corner coordinates
            assert engine._coord_to_gtp(0, 0) == "A9"
            assert engine._coord_to_gtp(8, 8) == "J1"
            assert engine._coord_to_gtp(4, 4) == "E5"

            # Test reverse conversion
            assert engine._gtp_to_coord("A9") == (0, 0)
            assert engine._gtp_to_coord("J1") == (8, 8)
            assert engine._gtp_to_coord("E5") == (4, 4)

            # Note: GTP skips 'I', so column 8 is 'J'
            assert engine._coord_to_gtp(0, 8) == "J9"
        finally:
            engine.close()
