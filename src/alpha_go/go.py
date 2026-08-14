"""Go game implementation with fast Python engine.

This module provides:
- FastGoBoard: Efficient numpy-based Go board for MCTS simulations
- GoState: MCTS-compatible state wrapper
- Constants: EMPTY, BLACK, WHITE

The implementation uses numpy arrays for cache-efficient board operations
and includes:
- Stone placement with capture detection
- Ko rule (simple ko)
- Suicide detection
- Liberty counting via flood fill
- Chinese rules scoring
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from alpha_go.engine import GTPEngine

# Go board constants
EMPTY = 0
BLACK = 1
WHITE = 2


class FastGoBoard:
    """Fast Python implementation of Go rules for MCTS rollouts.

    Uses numpy arrays for efficient board operations. Implements:
    - Stone placement with capture detection
    - Ko rule (simple ko only)
    - Suicide detection
    - Liberty counting via flood fill

    This is much faster than GTPEngine for MCTS simulations since it
    avoids subprocess overhead.
    """

    __slots__ = (
        "size",
        "board",
        "to_play",
        "ko_point",
        "passes",
        "move_count",
        "_neighbor_cache",
    )

    def __init__(self, size: int = 9) -> None:
        """Initialize empty board.

        Args:
            size: Board size (default 9)
        """
        self.size = size
        self.board: np.ndarray[Any, np.dtype[np.int8]] = np.zeros(
            (size, size), dtype=np.int8
        )
        self.to_play = BLACK
        self.ko_point: tuple[int, int] | None = None  # Illegal due to ko
        self.passes = 0  # Consecutive passes
        self.move_count = 0

        # Pre-compute neighbor offsets for efficiency
        self._neighbor_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r in range(size):
            for c in range(size):
                neighbors = []
                if r > 0:
                    neighbors.append((r - 1, c))
                if r < size - 1:
                    neighbors.append((r + 1, c))
                if c > 0:
                    neighbors.append((r, c - 1))
                if c < size - 1:
                    neighbors.append((r, c + 1))
                self._neighbor_cache[(r, c)] = neighbors

    def copy(self) -> "FastGoBoard":
        """Create a copy of this board state."""
        new_board = FastGoBoard.__new__(FastGoBoard)
        new_board.size = self.size
        new_board.board = self.board.copy()
        new_board.to_play = self.to_play
        new_board.ko_point = self.ko_point
        new_board.passes = self.passes
        new_board.move_count = self.move_count
        new_board._neighbor_cache = self._neighbor_cache  # Shared, immutable
        return new_board

    def _get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Get neighboring positions (cached for speed)."""
        return self._neighbor_cache[(row, col)]

    def _get_group_and_liberties(
        self, row: int, col: int
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Find all stones in a group and its liberties using flood fill.

        Args:
            row, col: Starting position (must contain a stone)

        Returns:
            group: Set of (row, col) positions in the group
            liberties: Set of (row, col) empty positions adjacent to group
        """
        color = self.board[row, col]
        if color == EMPTY:
            return set(), set()

        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for nr, nc in self._get_neighbors(r, c):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group, liberties

    def _remove_group(self, group: set[tuple[int, int]]) -> int:
        """Remove a group of stones from the board.

        Args:
            group: Set of positions to remove

        Returns:
            Number of stones removed (for scoring)
        """
        for r, c in group:
            self.board[r, c] = EMPTY
        return len(group)

    def _would_be_suicide(self, row: int, col: int, color: int) -> bool:
        """Check if placing a stone would be suicide (no liberties, no captures).

        Args:
            row, col: Position to check
            color: Color of stone to place

        Returns:
            True if the move would be suicide
        """
        opponent = WHITE if color == BLACK else BLACK

        # Temporarily place the stone
        self.board[row, col] = color

        # Check if we have liberties
        _, liberties = self._get_group_and_liberties(row, col)
        if liberties:
            self.board[row, col] = EMPTY
            return False

        # Check if we capture any opponent stones
        for nr, nc in self._get_neighbors(row, col):
            if self.board[nr, nc] == opponent:
                _, opp_liberties = self._get_group_and_liberties(nr, nc)
                if not opp_liberties:
                    # We capture something, not suicide
                    self.board[row, col] = EMPTY
                    return False

        self.board[row, col] = EMPTY
        return True

    def is_legal(self, row: int | None, col: int | None) -> bool:
        """Check if a move is legal.

        Args:
            row, col: Position to play, or None for pass

        Returns:
            True if the move is legal
        """
        # Pass is always legal
        if row is None or col is None:
            return True

        # Must be on board
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False

        # Must be empty
        if self.board[row, col] != EMPTY:
            return False

        # Can't play on ko point
        if self.ko_point == (row, col):
            return False

        # Can't be suicide
        if self._would_be_suicide(row, col, self.to_play):
            return False

        return True

    def play(self, row: int | None, col: int | None) -> bool:
        """Play a move.

        Args:
            row, col: Position to play, or None for pass

        Returns:
            True if move was played, False if illegal
        """
        # Handle pass
        if row is None or col is None:
            self.passes += 1
            self.to_play = WHITE if self.to_play == BLACK else BLACK
            self.ko_point = None
            self.move_count += 1
            return True

        if not self.is_legal(row, col):
            return False

        color = self.to_play
        opponent = WHITE if color == BLACK else BLACK

        # Place the stone
        self.board[row, col] = color
        self.passes = 0
        self.move_count += 1

        # Check for captures
        captured_count = 0
        captured_point: tuple[int, int] | None = None

        for nr, nc in self._get_neighbors(row, col):
            if self.board[nr, nc] == opponent:
                group, liberties = self._get_group_and_liberties(nr, nc)
                if not liberties:
                    if len(group) == 1:
                        captured_point = next(iter(group))
                    captured_count += self._remove_group(group)

        # Set ko point if exactly one stone captured
        if captured_count == 1 and captured_point is not None:
            # Check if this could be a ko (the capturing stone has exactly one liberty)
            _, my_liberties = self._get_group_and_liberties(row, col)
            if len(my_liberties) == 1:
                self.ko_point = captured_point
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        self.to_play = opponent
        return True

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves (excluding pass).

        Returns:
            List of (row, col) legal moves
        """
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c):
                    moves.append((r, c))
        return moves

    def is_game_over(self) -> bool:
        """Check if game is over (two consecutive passes)."""
        return self.passes >= 2

    def score(self) -> float:
        """Calculate score (Chinese rules: area scoring).

        Returns:
            Score from Black's perspective (positive = Black wins)
        """
        black_score = 0.0
        white_score = 6.5  # Komi

        # Count stones and territory
        counted = np.zeros((self.size, self.size), dtype=bool)

        for r in range(self.size):
            for c in range(self.size):
                if counted[r, c]:
                    continue

                cell = self.board[r, c]
                if cell == BLACK:
                    black_score += 1
                    counted[r, c] = True
                elif cell == WHITE:
                    white_score += 1
                    counted[r, c] = True
                else:
                    # Empty - flood fill to find territory
                    territory, borders = self._flood_empty(r, c, counted)
                    if borders == {BLACK}:
                        black_score += len(territory)
                    elif borders == {WHITE}:
                        white_score += len(territory)
                    # else: neutral territory, no points

        return black_score - white_score

    def _flood_empty(
        self, row: int, col: int, counted: "np.ndarray[Any, Any]"
    ) -> tuple[set[tuple[int, int]], set[int]]:
        """Flood fill empty region to determine territory ownership.

        Args:
            row, col: Starting empty position
            counted: Array tracking already-counted positions

        Returns:
            territory: Set of empty positions in this region
            borders: Set of colors bordering this region
        """
        territory: set[tuple[int, int]] = set()
        borders: set[int] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if counted[r, c]:
                continue
            if self.board[r, c] != EMPTY:
                borders.add(int(self.board[r, c]))
                continue

            counted[r, c] = True
            territory.add((r, c))

            for nr, nc in self._get_neighbors(r, c):
                if not counted[nr, nc]:
                    stack.append((nr, nc))

        return territory, borders

    def get_winner(self) -> int | None:
        """Get the winner of the game.

        Returns:
            BLACK (1) if black wins, WHITE (2) if white wins, None if draw
        """
        score = self.score()
        if score > 0:
            return BLACK
        elif score < 0:
            return WHITE
        return None


class GoState:
    """Go game state for MCTS using fast Python Go engine.

    Actions are (row, col) tuples or None for pass.
    Uses FastGoBoard for efficient game simulation without subprocess overhead.

    Implements the MCTS State protocol with:
    - get_legal_actions() -> list of actions
    - apply_action(action) -> new state
    - is_terminal() -> bool
    - get_reward(player) -> float
    - current_player() -> int
    - clone() -> copy of state
    """

    __slots__ = ("_go_board", "_legal_moves_cache")

    def __init__(self, go_board: FastGoBoard) -> None:
        """Initialize Go state from a FastGoBoard.

        Args:
            go_board: The underlying Go board state
        """
        self._go_board = go_board
        self._legal_moves_cache: list[tuple[int, int]] | None = None

    @classmethod
    def new_game(cls, size: int = 9) -> "GoState":
        """Create a new game state."""
        return cls(FastGoBoard(size))

    @classmethod
    def from_engine(cls, engine: "GTPEngine") -> "GoState":
        """Create GoState from a GTPEngine.

        Copies the board state from GTPEngine into a FastGoBoard.
        """
        board_list = engine.get_board()
        size = engine.size

        go_board = FastGoBoard(size)
        for r in range(size):
            for c in range(size):
                go_board.board[r, c] = board_list[r][c]
        go_board.to_play = engine.to_play

        return cls(go_board)

    @classmethod
    def from_cpp_board(cls, cpp_board: "Any") -> "GoState":
        """Create GoState from a C++ GoBoard.

        Copies the board state from C++ GoBoard into a FastGoBoard.
        """
        size = cpp_board.size()

        go_board = FastGoBoard(size)
        go_board.board = cpp_board.to_numpy().copy()
        go_board.to_play = cpp_board.to_play()

        return cls(go_board)

    @classmethod
    def from_board_array(
        cls,
        board: "np.ndarray[Any, Any] | list[list[int]]",
        to_play: int = BLACK,
        size: int = 9,
    ) -> "GoState":
        """Create GoState from a board array.

        Args:
            board: 2D array of board state (0=empty, 1=black, 2=white)
            to_play: Current player (BLACK=1 or WHITE=2)
            size: Board size
        """
        go_board = FastGoBoard(size)
        if isinstance(board, np.ndarray):
            go_board.board = board.astype(np.int8).copy()
        else:
            go_board.board = np.array(board, dtype=np.int8)
        go_board.to_play = to_play
        return cls(go_board)

    def get_legal_actions(self) -> list[tuple[int, int] | None]:
        """Return list of legal moves including pass."""
        if self._legal_moves_cache is None:
            self._legal_moves_cache = self._go_board.get_legal_moves()

        # Include pass
        actions: list[tuple[int, int] | None] = list(self._legal_moves_cache)
        actions.append(None)
        return actions

    def apply_action(self, action: tuple[int, int] | None) -> "GoState":
        """Apply action and return new state."""
        new_board = self._go_board.copy()

        if action is None:
            new_board.play(None, None)
        else:
            row, col = action
            new_board.play(row, col)

        return GoState(new_board)

    def is_terminal(self) -> bool:
        """Check if game is over (two consecutive passes)."""
        return self._go_board.is_game_over()

    def get_reward(self, player: int) -> float:
        """Get reward for player. Only valid at terminal state.

        Args:
            player: Player index (0 for BLACK, 1 for WHITE)

        Returns:
            1.0 if player wins, 0.0 if player loses
        """
        winner = self._go_board.get_winner()
        player_color = BLACK if player == 0 else WHITE

        if winner == player_color:
            return 1.0
        elif winner is None:
            return 0.5  # Draw
        return 0.0

    def current_player(self) -> int:
        """Return current player index (0 for BLACK, 1 for WHITE)."""
        if self._go_board.to_play == BLACK:
            return 0
        return 1

    def clone(self) -> "GoState":
        """Return a deep copy of this state."""
        return GoState(self._go_board.copy())

    @property
    def board(self) -> list[list[int]]:
        """Get the board as a list of lists (for compatibility)."""
        return cast(list[list[int]], self._go_board.board.tolist())

    @property
    def board_array(self) -> "np.ndarray[Any, Any]":
        """Get the board as a numpy array."""
        return self._go_board.board

    @property
    def to_play(self) -> int:
        """Get current player to move."""
        return self._go_board.to_play

    @property
    def board_size(self) -> int:
        """Get board size."""
        return self._go_board.size
