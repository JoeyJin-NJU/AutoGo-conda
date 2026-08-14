"""Tests for C++ Go game implementation parity with Python."""

import numpy as np
import pytest

import alpha_go_cpp
from alpha_go.go import FastGoBoard, BLACK, WHITE, EMPTY


@pytest.fixture
def cpp_board():
    """Create a C++ GoBoard."""
    return alpha_go_cpp.GoBoard(9)


@pytest.fixture
def py_board():
    """Create a Python FastGoBoard."""
    return FastGoBoard(size=9)


class TestGoBoardBasics:
    """Test basic GoBoard functionality."""

    def test_construction(self, cpp_board, py_board):
        """Test board construction."""
        assert cpp_board.size() == py_board.size
        assert cpp_board.to_play() == alpha_go_cpp.GoBoard.BLACK
        assert py_board.to_play == BLACK

    def test_legal_moves_initial(self, cpp_board, py_board):
        """Test legal moves on empty board."""
        cpp_moves = set(cpp_board.get_legal_moves_flat())
        py_moves = {r * 9 + c for r, c in py_board.get_legal_moves()}

        # Both should have 81 legal moves (9x9)
        assert len(cpp_moves) == 81
        assert len(py_moves) == 81

    def test_play_single_stone(self, cpp_board, py_board):
        """Test playing a single stone."""
        # Play at (3, 3)
        assert cpp_board.play(3, 3) == True
        assert py_board.play(3, 3) == True

        # Check stone placement
        assert cpp_board.at(3, 3) == alpha_go_cpp.GoBoard.BLACK
        assert py_board.board[3, 3] == BLACK

        # Check turn switched
        assert cpp_board.to_play() == alpha_go_cpp.GoBoard.WHITE
        assert py_board.to_play == WHITE

    def test_play_multiple_stones(self, cpp_board, py_board):
        """Test playing multiple stones."""
        moves = [(3, 3), (4, 4), (3, 4), (4, 3)]

        for row, col in moves:
            cpp_board.play(row, col)
            py_board.play(row, col)

        # Compare board states
        cpp_arr = cpp_board.to_numpy()
        for row in range(9):
            for col in range(9):
                # Convert Python values (-1, 0, 1) to C++ values (0, 1, 2)
                py_val = py_board.board[row, col]
                if py_val == EMPTY:
                    expected = 0  # EMPTY
                elif py_val == BLACK:
                    expected = 1  # BLACK
                else:
                    expected = 2  # WHITE
                assert cpp_arr[row, col] == expected, f"Mismatch at ({row}, {col})"


class TestCapture:
    """Test capture functionality."""

    def test_simple_capture(self, cpp_board, py_board):
        """Test capturing a single stone."""
        # Create a capture situation:
        # . B .
        # B W B
        # . B .
        # Where W at (4,4) is surrounded

        # Alternating play to set up capture:
        # 1. Black (3,4), 2. White (4,4), 3. Black (5,4), 4. White passes
        # 5. Black (4,3), 6. White passes, 7. Black (4,5) captures

        cpp_board.play(3, 4)  # Black
        py_board.play(3, 4)
        cpp_board.play(4, 4)  # White - will be surrounded
        py_board.play(4, 4)
        cpp_board.play(5, 4)  # Black
        py_board.play(5, 4)
        cpp_board.pass_move()  # White passes
        py_board.play(None, None)
        cpp_board.play(4, 3)  # Black
        py_board.play(4, 3)
        cpp_board.pass_move()  # White passes
        py_board.play(None, None)
        cpp_board.play(4, 5)  # Black captures
        py_board.play(4, 5)

        # White stone should be captured
        assert cpp_board.at(4, 4) == alpha_go_cpp.GoBoard.EMPTY
        assert py_board.board[4, 4] == EMPTY


class TestSuicide:
    """Test suicide rule implementation."""

    def test_single_stone_suicide_illegal(self, cpp_board):
        """Test that single stone suicide is illegal.

        Setup (Black to move):
          . B .
          B . B
          . B .

        Playing at center would be suicide.
        """
        # White passes, then sets up surrounding
        cpp_board.pass_move()  # White
        cpp_board.play(3, 4)   # Black
        cpp_board.pass_move()  # White
        cpp_board.play(4, 3)   # Black
        cpp_board.pass_move()  # White
        cpp_board.play(4, 5)   # Black
        cpp_board.pass_move()  # White
        cpp_board.play(5, 4)   # Black
        # Now it's White's turn and (4,4) is surrounded by Black

        # White cannot play at (4,4) - suicide
        assert cpp_board.is_legal(4, 4) == False
        assert cpp_board.play(4, 4) == False
        assert (4 * 9 + 4) not in cpp_board.get_legal_moves_flat()

    def test_multi_stone_suicide_illegal(self, cpp_board):
        """Test that multi-stone suicide is illegal.

        This replicates the bug found during game with seed=30.
        Setup: A single black stone at H6 is surrounded by white on 3 sides,
        with only J6 as its liberty. Playing at J6 would connect H6 but
        the combined group would have 0 liberties = suicide.

        Board visualization (9x9, J6 = row 3, col 8):
               A B C D E F G H J
            9  . . . . . . . W B
            8  . . . . . . . W B
            7  . . . . . . W W B
            6  . . . . . W W B .  <- H6 has black, J6 empty
            5  . . . . . . W W W
            4  . . . . . . W . W
            ...
        """
        # Set up the board state that triggers multi-stone suicide
        # We'll manually construct a simpler version:
        # Black stone at H6 (row=3, col=7) surrounded by white except J6 (row=3, col=8)
        # Then black chain at J7-J8-J9 (row 2,1,0 col 8) also surrounded

        # Simpler setup: H6 black stone with J6 as only liberty,
        # surrounded by white at G6, H5, H7
        # And J5, J7 also white so playing J6 gives the group 0 liberties

        board = alpha_go_cpp.GoBoard(9)

        # Place white stones to surround H6 and block J6's potential liberties
        # H7 (row=2, col=7) - white
        # G6 (row=3, col=6) - white
        # H5 (row=4, col=7) - white
        # J7 (row=2, col=8) - white
        # J5 (row=4, col=8) - white

        moves = [
            # Black, White alternating - setting up the trap
            (0, 0),  # Black irrelevant
            (2, 7),  # White H7
            (0, 1),  # Black irrelevant
            (3, 6),  # White G6
            (0, 2),  # Black irrelevant
            (4, 7),  # White H5
            (0, 3),  # Black irrelevant
            (2, 8),  # White J7
            (0, 4),  # Black irrelevant
            (4, 8),  # White J5
            (3, 7),  # Black H6 - the stone that will be trapped
        ]

        for row, col in moves:
            board.play(row, col)

        # Now it's White's turn. Skip to make it Black's turn again.
        board.pass_move()  # White passes

        # Current state:
        # H6 (row=3, col=7) has Black
        # H6's neighbors: H7(W), G6(W), H5(W), J6(empty)
        # H6's only liberty is J6
        # If Black plays J6, the H6+J6 group has neighbors:
        #   H7(W), G6(W), H5(W), J7(W), J5(W) - all white!
        # So the group would have 0 liberties = multi-stone suicide

        # Verify H6 has black stone
        assert board.at(3, 7) == alpha_go_cpp.GoBoard.BLACK

        # Verify J6 is empty
        assert board.at(3, 8) == alpha_go_cpp.GoBoard.EMPTY

        # Verify surrounding white stones
        assert board.at(2, 7) == alpha_go_cpp.GoBoard.WHITE  # H7
        assert board.at(3, 6) == alpha_go_cpp.GoBoard.WHITE  # G6
        assert board.at(4, 7) == alpha_go_cpp.GoBoard.WHITE  # H5
        assert board.at(2, 8) == alpha_go_cpp.GoBoard.WHITE  # J7
        assert board.at(4, 8) == alpha_go_cpp.GoBoard.WHITE  # J5

        # Now Black tries to play J6 - should be illegal (multi-stone suicide)
        assert board.to_play() == alpha_go_cpp.GoBoard.BLACK
        assert board.is_legal(3, 8) == False, "J6 should be illegal - multi-stone suicide"
        assert board.play(3, 8) == False, "Playing J6 should fail"

        # J6 should NOT be in legal moves
        legal_moves = board.get_legal_moves_flat()
        j6_flat = 3 * 9 + 8  # row=3, col=8
        assert j6_flat not in legal_moves, "J6 should not be in legal moves"


class TestKoRule:
    """Test Ko rule implementation."""

    def test_ko_detection(self, cpp_board, py_board):
        """Test that Ko rule is enforced."""
        # Create a Ko situation
        # This is a simplified test - full Ko requires specific setup
        pass  # TODO: Implement detailed Ko test


class TestGameEnd:
    """Test game ending conditions."""

    def test_two_passes(self, cpp_board, py_board):
        """Test that two passes end the game."""
        cpp_board.pass_move()
        cpp_board.pass_move()
        assert cpp_board.is_game_over() == True

        py_board.play(None, None)  # pass
        py_board.play(None, None)  # pass
        assert py_board.is_game_over() == True


class TestNumpy:
    """Test numpy conversion."""

    def test_to_numpy(self, cpp_board):
        """Test numpy array export."""
        cpp_board.play(3, 3)
        cpp_board.play(4, 4)

        arr = cpp_board.to_numpy()
        assert arr.shape == (9, 9)
        assert arr.dtype == np.int8
        assert arr[3, 3] == 1  # BLACK
        assert arr[4, 4] == 2  # WHITE


class TestCopy:
    """Test board copying."""

    def test_copy(self, cpp_board):
        """Test that copy creates independent board."""
        cpp_board.play(3, 3)
        copy = cpp_board.copy()

        # Modify original
        cpp_board.play(4, 4)

        # Copy should be unchanged
        assert copy.at(4, 4) == alpha_go_cpp.GoBoard.EMPTY
        assert cpp_board.at(4, 4) == alpha_go_cpp.GoBoard.WHITE

    def test_py_copy(self, py_board):
        """Test that Python copy creates independent board."""
        py_board.play(3, 3)
        copy = py_board.copy()

        # Modify original
        py_board.play(4, 4)

        # Copy should be unchanged
        assert copy.board[4, 4] == EMPTY
        assert py_board.board[4, 4] == WHITE


class TestCppOnlyGoBoard:
    """Standalone tests for C++ GoBoard (no Python parity)."""

    def test_basic_game_flow(self, cpp_board):
        """Test basic game flow."""
        # Play some moves
        assert cpp_board.play(0, 0) == True  # Black
        assert cpp_board.play(0, 1) == True  # White
        assert cpp_board.play(1, 0) == True  # Black
        assert cpp_board.play(1, 1) == True  # White

        # Check positions
        assert cpp_board.at(0, 0) == alpha_go_cpp.GoBoard.BLACK
        assert cpp_board.at(0, 1) == alpha_go_cpp.GoBoard.WHITE
        assert cpp_board.at(1, 0) == alpha_go_cpp.GoBoard.BLACK
        assert cpp_board.at(1, 1) == alpha_go_cpp.GoBoard.WHITE

    def test_illegal_move_occupied(self, cpp_board):
        """Test that occupied positions are illegal."""
        cpp_board.play(3, 3)
        # Can't play on occupied position
        assert cpp_board.is_legal(3, 3) == False

    def test_pass_and_game_over(self, cpp_board):
        """Test passing and game ending."""
        assert cpp_board.is_game_over() == False
        cpp_board.pass_move()
        assert cpp_board.is_game_over() == False
        cpp_board.pass_move()
        assert cpp_board.is_game_over() == True

    def test_row_col_conversion(self, cpp_board):
        """Test flat index to row/col conversion."""
        row, col = cpp_board.row_col(30)  # 3*9 + 3
        assert row == 3
        assert col == 3

    def test_move_count(self, cpp_board):
        """Test move counting."""
        assert cpp_board.move_count() == 0
        cpp_board.play(3, 3)
        assert cpp_board.move_count() == 1
        cpp_board.pass_move()
        assert cpp_board.move_count() == 2

    def test_capture_corner(self, cpp_board):
        """Test capturing a corner stone."""
        # Place white stone in corner
        cpp_board.pass_move()  # Black passes
        cpp_board.play(0, 0)   # White at corner

        # Surround with black
        cpp_board.play(0, 1)   # Black
        cpp_board.pass_move()  # White passes
        cpp_board.play(1, 0)   # Black captures

        # Corner should be empty now
        assert cpp_board.at(0, 0) == alpha_go_cpp.GoBoard.EMPTY

    def test_score_empty_board(self, cpp_board):
        """Test scoring empty board."""
        cpp_board.pass_move()
        cpp_board.pass_move()
        # Empty board: all territory is dead space (or divided equally)
        # Implementation may vary
        score = cpp_board.score()
        # Score should be a reasonable value
        assert isinstance(score, float)


class TestPythonOnlyFastGoBoard:
    """Standalone tests for Python FastGoBoard (no C++ parity)."""

    def test_basic_game_flow(self, py_board):
        """Test basic game flow."""
        assert py_board.play(0, 0) == True  # Black
        assert py_board.play(0, 1) == True  # White
        assert py_board.play(1, 0) == True  # Black
        assert py_board.play(1, 1) == True  # White

        assert py_board.board[0, 0] == BLACK
        assert py_board.board[0, 1] == WHITE
        assert py_board.board[1, 0] == BLACK
        assert py_board.board[1, 1] == WHITE

    def test_illegal_move_occupied(self, py_board):
        """Test that occupied positions are illegal."""
        py_board.play(3, 3)
        assert py_board.is_legal(3, 3) == False

    def test_pass_and_game_over(self, py_board):
        """Test passing and game ending."""
        assert py_board.is_game_over() == False
        py_board.play(None, None)
        assert py_board.is_game_over() == False
        py_board.play(None, None)
        assert py_board.is_game_over() == True

    def test_move_count(self, py_board):
        """Test move counting."""
        assert py_board.move_count == 0
        py_board.play(3, 3)
        assert py_board.move_count == 1
        py_board.play(None, None)
        assert py_board.move_count == 2

    def test_capture_corner(self, py_board):
        """Test capturing a corner stone."""
        # Place white stone in corner
        py_board.play(None, None)  # Black passes
        py_board.play(0, 0)        # White at corner

        # Surround with black
        py_board.play(0, 1)        # Black
        py_board.play(None, None)  # White passes
        py_board.play(1, 0)        # Black captures

        # Corner should be empty now
        assert py_board.board[0, 0] == EMPTY

    def test_score_empty_board(self, py_board):
        """Test scoring empty board."""
        py_board.play(None, None)
        py_board.play(None, None)
        score = py_board.score()
        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
