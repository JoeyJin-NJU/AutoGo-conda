"""GTP Engine wrapper for GNU Go."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Literal

import alpha_go_cpp

# Constants
BLACK = 1
WHITE = 2
EMPTY = 0

COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips 'I'


@dataclass
class GTPEngine:
    """Wrapper for GNU Go GTP engine."""

    size: int
    process: subprocess.Popen[str] = field(repr=False)
    to_play: Literal[1, 2] = BLACK  # 1=BLACK, 2=WHITE
    last_move: tuple[int, int] | None = None
    consecutive_passes: int = 0
    _is_over: bool = False
    _result: str | None = None
    move_history: list[tuple[int, int] | None] = field(default_factory=list)

    @classmethod
    def new(cls, size: int = 9, level: int = 1) -> GTPEngine:
        """Start a new GNU Go process."""
        gnugo_bin = shutil.which("gnugo") or "/usr/games/gnugo"
        process = subprocess.Popen(
            [gnugo_bin, "--mode", "gtp", "--level", str(level)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        engine = cls(size=size, process=process)
        engine._send(f"boardsize {size}")
        engine._send("clear_board")
        engine._send(f"komi {alpha_go_cpp.GoBoard.KOMI}")
        return engine

    def _send(self, command: str) -> str:
        """Send a GTP command and return the response."""
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if line.strip() == "":
                break
            response_lines.append(line)

        response = "".join(response_lines).strip()
        if response.startswith("?"):
            raise ValueError(response[1:].strip())  # Error response
        return response.lstrip("= ").strip()

    def _coord_to_gtp(self, row: int, col: int) -> str:
        """Convert (row, col) to GTP coordinate like 'D4'."""
        return f"{COLS[col]}{self.size - row}"

    def _gtp_to_coord(self, gtp: str) -> tuple[int, int] | None:
        """Convert GTP coordinate like 'D4' to (row, col)."""
        if not gtp or gtp.upper() in ("PASS", "RESIGN"):
            return None
        gtp = gtp.upper()
        col = COLS.index(gtp[0])
        row = self.size - int(gtp[1:])
        return (row, col)

    def get_board(self) -> list[list[int]]:
        """Get current board state as 2D list."""
        board = [[EMPTY] * self.size for _ in range(self.size)]

        # Use list_stones to get positions
        for color, value in [("black", BLACK), ("white", WHITE)]:
            response = self._send(f"list_stones {color}")
            if response:
                for stone in response.split():
                    coord = self._gtp_to_coord(stone)
                    if coord:
                        board[coord[0]][coord[1]] = value

        return board

    def is_legal(self, row: int, col: int) -> bool:
        """Check if a move is legal."""
        color = "black" if self.to_play == BLACK else "white"
        gtp_coord = self._coord_to_gtp(row, col)
        response = self._send(f"is_legal {color} {gtp_coord}")
        return response == "1"

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves for current player."""
        color = "black" if self.to_play == BLACK else "white"
        response = self._send(f"all_legal {color}")
        moves = []
        if response:
            for move in response.split():
                coord = self._gtp_to_coord(move)
                if coord:
                    moves.append(coord)
        return moves

    def play(self, row: int | None, col: int | None) -> bool:
        """Play a move. Returns True if successful."""
        color = "black" if self.to_play == BLACK else "white"

        if row is None or col is None:
            # Pass
            self._send(f"play {color} pass")
            self.last_move = None
            self.consecutive_passes += 1
            self.move_history.append(None)
        else:
            gtp_coord = self._coord_to_gtp(row, col)
            try:
                self._send(f"play {color} {gtp_coord}")
            except ValueError:
                return False  # Illegal move
            self.last_move = (row, col)
            self.consecutive_passes = 0
            self.move_history.append((row, col))

        # Switch turn
        self.to_play = WHITE if self.to_play == BLACK else BLACK

        # Check if game is over (two passes)
        if self.consecutive_passes >= 2:
            self._is_over = True
            self._result = self._get_final_score()

        return True

    def undo(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.move_history:
            return False

        try:
            self._send("undo")
        except ValueError:
            return False

        # Remove last move from history
        self.move_history.pop()

        # Switch turn back
        self.to_play = WHITE if self.to_play == BLACK else BLACK

        # Update last_move to previous move (or None if no moves left)
        self.last_move = self.move_history[-1] if self.move_history else None

        # Reset consecutive passes (recalculate from recent history)
        self.consecutive_passes = 0
        for move in reversed(self.move_history):
            if move is None:
                self.consecutive_passes += 1
            else:
                break

        # Reset game over state
        self._is_over = False
        self._result = None

        return True

    def genmove(self) -> tuple[int, int] | None:
        """Generate and play a move for the current player."""
        color = "black" if self.to_play == BLACK else "white"
        response = self._send(f"genmove {color}")

        if response.upper() in ("PASS", "RESIGN"):
            self.last_move = None
            self.consecutive_passes += 1
            self.to_play = WHITE if self.to_play == BLACK else BLACK

            if self.consecutive_passes >= 2 or response.upper() == "RESIGN":
                self.end_game()
            return None

        coord = self._gtp_to_coord(response)
        if coord:
            self.last_move = coord
            self.consecutive_passes = 0
            self.to_play = WHITE if self.to_play == BLACK else BLACK

        return coord

    def end_game(self):
        # invoke this in eval if game is decided to be over, e.g. max_steps reached
        self._is_over = True
        self._result = self._get_final_score()

    def suggest_move(self, seed: int) -> tuple[int, int] | None:
        """Generate a move suggestion without playing it."""
        color = "black" if self.to_play == BLACK else "white"
        response = self._send(f"gg_genmove {color} {seed}")
        if response.upper() in ("PASS", "RESIGN"):
            return None
        return self._gtp_to_coord(response)

    def _get_final_score(self) -> str:
        """Get final score from GNU Go."""
        response = self._send("final_score")
        return response if response else "?"

    def is_over(self) -> bool:
        """Check if game is over."""
        return self._is_over

    def result(self) -> str | None:
        """Get game result."""
        return self._result

    def close(self) -> None:
        """Close the GNU Go process."""
        self._send("quit")
        self.process.terminate()
        self.process.wait()
