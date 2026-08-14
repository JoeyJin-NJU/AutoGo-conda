"""Consolidated gameplay module for playing Go games between agents.

This module provides a unified interface for playing games, collecting data,
and logging results. It consolidates functionality from self_play.py and
collector.py into a single, configurable implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from alpha_go.agents import Agent, PASS, RESIGN
from alpha_go.analysis.plotting import render_board_simple, get_column_labels

# Re-export constants from C++ module for compatibility
BLACK = alpha_go_cpp.GoBoard.BLACK
WHITE = alpha_go_cpp.GoBoard.WHITE

console = Console()


@dataclass
class MoveMetric:
    """Debug metrics for a single move."""
    move_step: int
    policy_entropy: float | None
    agent_color: str  # "black" or "white"
    # MCTS search statistics (None if agent is not MCTS-based)
    is_teacher: bool | None = None  # This move should be used to supervise the policy network
    visit_counts: NDArray[Any] | None = None   # (board_size^2+1,) int16 dense
    q_values: NDArray[Any] | None = None       # (board_size^2+1,) float32 dense
    # Root policy priors fed into MCTS (post Dirichlet noise if any). Stored as a
    # dense distribution over actions so we can compute argmax(policy) without
    # re-running the network. NaN for actions not in policy_dict (illegal moves).
    policy_priors: NDArray[Any] | None = None  # (board_size^2+1,) float32 dense
    temperature: float | None = None
    root_value: float | None = None            # root Q after all simulations (current-player win prob)
    # Raw NN value at each root-child at expansion time (teacher-mode analysis).
    # Same perspective as q_values; NaN for actions never expanded.
    first_eval_values: NDArray[Any] | None = None  # (board_size^2+1,) float32 dense
    # Deepest depth reached in the subtree under each root child (teacher-mode).
    # 0 for actions never expanded.
    max_subtree_depths: NDArray[Any] | None = None  # (board_size^2+1,) int16 dense


@dataclass
class GameRecord:
    """Record of a single game.

    Attributes:
        board_size: Size of the board (9, 13, or 19).
        black_agent: Name of the black agent.
        white_agent: Name of the white agent.
        moves: List of moves played. Each move is (row, col) or PASS=(-1,-1).
        boards: List of board states before each move.
        move_metrics: Optional list of move metrics (entropy, etc.).
        winner: BLACK, WHITE, or None for draw.
        result: Human-readable result string (e.g., "B+2.5").
        num_moves: Number of moves played.
        komi: Komi used for scoring.
        termination: How the game ended ("double_pass" or "max_moves").
        black_checkpoint_path: Optional checkpoint path for black agent.
        white_checkpoint_path: Optional checkpoint path for white agent.
    """
    board_size: int
    black_agent: str
    white_agent: str
    moves: list[tuple[int, int]] = field(default_factory=list)
    boards: list[NDArray[Any]] = field(default_factory=list)
    move_metrics: list[MoveMetric] = field(default_factory=list)
    winner: int | None = None
    result: str = ""
    num_moves: int = 0
    komi: float = alpha_go_cpp.GoBoard.KOMI
    termination: str = ""  # "double_pass" or "max_moves"
    black_checkpoint_path: str | None = None
    white_checkpoint_path: str | None = None
    # Wall-clock seconds spent inside black/white agent select_move calls,
    # and the number of moves each side made (used to derive sec/move).
    black_move_seconds: float = 0.0
    white_move_seconds: float = 0.0
    black_move_count: int = 0
    white_move_count: int = 0


def render_illegal_move_debug(
    board: alpha_go_cpp.GoBoard,
    invalid_move: tuple[int, int],
    move_count: int,
    seed: int,
    moves_so_far: list[tuple[int, int]],
) -> str:
    """Render board with invalid move marked and save to /tmp/.

    Args:
        board: Current board state
        invalid_move: The (row, col) that caused the error
        move_count: Current move number
        seed: Game seed
        moves_so_far: List of moves played so far

    Returns:
        Path to saved image
    """
    board_size = board.size()
    board_np = board.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    render_board_simple(board_np, ax, board_size=board_size)

    # Mark the invalid move with a red X
    row, col = invalid_move
    ax.plot(col, row, "rx", markersize=20, markeredgewidth=4)

    # Add a red circle around the invalid move position
    circle = mpatches.Circle((col, row), 0.45, fill=False, edgecolor="red", linewidth=3)
    ax.add_patch(circle)

    # Convert to Go notation for the title
    col_labels = get_column_labels(board_size)
    go_notation = f"{col_labels[col]}{board_size - row}"
    to_play = "Black" if board.to_play() == BLACK else "White"

    title = (
        f"IllegalMoveError: {to_play} tried to play at {go_notation} ({row}, {col})\n"
        f"Move #{move_count}, Seed: {seed}"
    )
    ax.set_title(title, fontsize=12, color="red")

    # Add move history as text below the board
    if moves_so_far:
        move_strs = []
        for i, (r, c) in enumerate(moves_so_far[-10:]):  # Last 10 moves
            if (r, c) == (-1, -1):
                move_strs.append(f"{i+max(0, len(moves_so_far)-10)+1}:pass")
            else:
                move_strs.append(f"{i+max(0, len(moves_so_far)-10)+1}:{col_labels[c]}{board_size - r}")
        fig.text(0.5, 0.02, "Last moves: " + ", ".join(move_strs), ha="center", fontsize=9)

    # Save to /tmp/
    timestamp = int(time.time())
    filepath = f"/tmp/illegal_move_debug_{timestamp}_seed{seed}_move{move_count}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def play_game(
    black_agent: Agent,
    white_agent: Agent,
    board_size: int = 9,
    max_moves: int = 150,
    seed: int = 0,
    collect_boards: bool = True,
    collect_metrics: bool = False,
    render_debug_on_error: bool = True,
    komi: float = alpha_go_cpp.GoBoard.KOMI,
    fallback_to_pass: bool = False,
    black_agent_name: str = "",
    white_agent_name: str = "",
    black_is_teacher: bool = False,
    white_is_teacher: bool = False,
) -> GameRecord:
    """Play a single game between two agents.

    Args:
        black_agent: Agent playing black stones.
        white_agent: Agent playing white stones.
        board_size: Size of the board (default 9).
        max_moves: Maximum moves before termination (default 150).
        seed: Random seed for reproducibility.
        collect_boards: Store board states for training data (default True).
        collect_metrics: Track entropy per move (default False).
        render_debug_on_error: Save debug images on illegal moves (default True).
        fallback_to_pass: Fall back to pass on illegal moves instead of raising (default False).
        black_agent_name: Override name for black agent (defaults to class name).
        white_agent_name: Override name for white agent (defaults to class name).

    Returns:
        GameRecord with game data.

    Raises:
        RuntimeError: If an illegal move is played and fallback_to_pass is False.
    """
    board = alpha_go_cpp.GoBoard(board_size, komi)

    # Get agent names and checkpoint paths
    black_name = black_agent_name or type(black_agent).__name__
    white_name = white_agent_name or type(white_agent).__name__
    black_ckpt = getattr(black_agent, 'checkpoint_path', None)
    white_ckpt = getattr(white_agent, 'checkpoint_path', None)

    record = GameRecord(
        board_size=board_size,
        black_agent=black_name,
        white_agent=white_name,
        black_checkpoint_path=black_ckpt,
        white_checkpoint_path=white_ckpt,
        komi=komi,
    )

    # Initialize agents
    black_agent.start_game(board_size)
    white_agent.start_game(board_size)

    try:
        agents = {BLACK: black_agent, WHITE: white_agent}
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            if collect_boards:
                record.boards.append(board.to_numpy().copy())

            current_player = board.to_play()
            current_agent = agents[current_player]

            _t0 = time.perf_counter()
            move = current_agent.select_move(board, seed + move_count)
            _dt = time.perf_counter() - _t0
            if current_player == BLACK:
                record.black_move_seconds += _dt
                record.black_move_count += 1
            else:
                record.white_move_seconds += _dt
                record.white_move_count += 1

            # Capture metrics if requested
            if collect_metrics:
                entropy = None
                visit_counts = None
                q_values = None
                policy_priors = None
                temperature = None
                root_value = None

                if hasattr(current_agent, "get_last_entropy"):
                    entropy = current_agent.get_last_entropy()

                # Extract MCTS search statistics if available
                search_result = getattr(current_agent, "last_search_result", None)
                if search_result is not None:
                    bs = board.size()
                    n_actions = bs * bs + 1
                    pass_idx = n_actions - 1

                    visit_counts = np.zeros(n_actions, dtype=np.int16)
                    for flat_idx, count in search_result.tree.get_child_visit_counts().items():
                        if flat_idx == alpha_go_cpp.PASS_ACTION:
                            visit_counts[pass_idx] = count
                        else:
                            visit_counts[flat_idx] = count

                    q_values = np.zeros(n_actions, dtype=np.float32)
                    for flat_idx, q in search_result.tree.get_child_q_values().items():
                        if flat_idx == alpha_go_cpp.PASS_ACTION:
                            q_values[pass_idx] = q
                        else:
                            q_values[flat_idx] = q

                    policy_priors = np.zeros(n_actions, dtype=np.float32)
                    for flat_idx, p in search_result.tree.get_root_policy_priors().items():
                        if flat_idx == alpha_go_cpp.PASS_ACTION:
                            policy_priors[pass_idx] = p
                        else:
                            policy_priors[flat_idx] = p

                    temperature = getattr(current_agent, "temperature", None)
                    # get_root_q_value() is from player_at_parent (opponent)
                    # perspective; flip so stored value = current-player win prob.
                    root_value = 1.0 - float(search_result.tree.get_root_q_value())

                record.move_metrics.append(MoveMetric(
                    move_step=move_count,
                    policy_entropy=entropy,
                    agent_color="black" if current_player == BLACK else "white",
                    is_teacher=black_is_teacher if current_player == BLACK else white_is_teacher,
                    visit_counts=visit_counts,
                    q_values=q_values,
                    policy_priors=policy_priors,
                    temperature=temperature,
                    root_value=root_value,
                ))

            if move == RESIGN:
                # The position and metrics above are recorded in anticipation
                # of a playable policy action. Resignation is not part of the
                # model action space and is therefore not appended to moves;
                # discard its otherwise-unpaired training row as well.
                if collect_boards:
                    record.boards.pop()
                if collect_metrics:
                    record.move_metrics.pop()

                # Opponent wins by resignation; score the current position
                record.num_moves = move_count
                record.termination = "resign"
                score = board.score()
                opponent = WHITE if current_player == BLACK else BLACK
                record.winner = opponent
                margin = abs(score)
                record.result = f"{'W' if opponent == WHITE else 'B'}+{margin:.1f}"
                break

            if move == PASS:
                board.pass_move()
            else:
                if board.is_legal(move[0], move[1]):
                    board.play(move[0], move[1])
                elif fallback_to_pass:
                    # Fall back to pass if move is illegal
                    board.pass_move()
                    move = PASS
                else:
                    # Illegal move - render debug and raise
                    if render_debug_on_error:
                        debug_path = render_illegal_move_debug(
                            board=board,
                            invalid_move=move,
                            move_count=move_count,
                            seed=seed,
                            moves_so_far=record.moves,
                        )
                        console.print(f"[red]Illegal move at move {move_count}[/red]")
                        console.print(f"[red]Invalid move: {move} (row={move[0]}, col={move[1]})[/red]")
                        console.print(f"[yellow]Debug image saved to: {debug_path}[/yellow]")
                    raise RuntimeError(f"Illegal move {move} at move {move_count}")

            # Notify both agents of the move
            try:
                black_agent.notify_move(move[0], move[1])
                white_agent.notify_move(move[0], move[1])
            except Exception as e:
                if render_debug_on_error:
                    debug_path = render_illegal_move_debug(
                        board=board,
                        invalid_move=move,
                        move_count=move_count,
                        seed=seed,
                        moves_so_far=record.moves,
                    )
                    console.print(f"[red]Error in notify_move at move {move_count}[/red]")
                    console.print(f"[red]Move: {move} (row={move[0]}, col={move[1]})[/red]")
                    console.print(f"[yellow]Debug image saved to: {debug_path}[/yellow]")
                raise

            record.moves.append(move)
            move_count += 1

        # Score the game (unless already decided by resignation)
        if record.termination != "resign":
            record.num_moves = move_count
            record.termination = "double_pass" if board.is_game_over() else "max_moves"
            score = board.score()
            if score > 0:
                record.result = f"B+{score:.1f}"
                record.winner = BLACK
            elif score < 0:
                record.result = f"W+{-score:.1f}"
                record.winner = WHITE
            else:
                record.result = "Draw"

    finally:
        black_agent.end_game()
        white_agent.end_game()

    return record


def save_game_data(
    record: GameRecord,
    output_dir: Path,
    game_index: int,
    date_slug: str,
) -> Path:
    """Save a single game record to an npz file.

    Args:
        record: GameRecord to save.
        output_dir: Directory to save the file.
        game_index: Index of the game (for filename).
        date_slug: Date slug for filename prefix.

    Returns:
        Path to the saved file.
    """
    filename = f"{date_slug}-game{game_index:07d}.npz"
    filepath = output_dir / filename

    boards = np.array(record.boards, dtype=np.int8) if record.boards else np.array([], dtype=np.int8)
    moves = np.array(record.moves, dtype=np.int16)

    save_kwargs: dict[str, Any] = dict(
        boards=boards,
        moves=moves,
        winner=record.winner if record.winner else 0,
        result=record.result,
        board_size=record.board_size,
        black_agent=record.black_agent,
        white_agent=record.white_agent,
        black_checkpoint_path=record.black_checkpoint_path or "",
        white_checkpoint_path=record.white_checkpoint_path or "",
        num_moves=record.num_moves,
        komi=record.komi,
        termination=record.termination,
        code_version="v5-mcts-stats",
    )

    # Add MCTS search statistics if available
    has_mcts = (
        record.move_metrics
        and any(m.visit_counts is not None for m in record.move_metrics)
    )
    if has_mcts:
        n_moves = len(record.move_metrics)
        n_actions = len(record.move_metrics[0].visit_counts) if record.move_metrics[0].visit_counts is not None else record.board_size ** 2 + 1

        all_visits = np.zeros((n_moves, n_actions), dtype=np.int16)
        all_q = np.zeros((n_moves, n_actions), dtype=np.float32)
        all_priors = np.zeros((n_moves, n_actions), dtype=np.float32)
        all_temps = np.zeros(n_moves, dtype=np.float32)
        all_root_values = np.zeros(n_moves, dtype=np.float32)
        all_is_teacher = np.zeros(n_moves, dtype=np.bool_)
        all_first_eval = np.zeros((n_moves, n_actions), dtype=np.float32)
        all_max_depth = np.zeros((n_moves, n_actions), dtype=np.int16)

        for i, m in enumerate(record.move_metrics):
            if m.visit_counts is not None:
                all_visits[i] = m.visit_counts
            if m.q_values is not None:
                all_q[i] = m.q_values
            if m.policy_priors is not None:
                all_priors[i] = m.policy_priors
            if m.temperature is not None:
                all_temps[i] = m.temperature
            if m.root_value is not None:
                all_root_values[i] = m.root_value
            if m.is_teacher:
                all_is_teacher[i] = True
            if m.first_eval_values is not None:
                all_first_eval[i] = m.first_eval_values
            if m.max_subtree_depths is not None:
                all_max_depth[i] = m.max_subtree_depths

        save_kwargs["mcts_visits"] = all_visits
        save_kwargs["mcts_q_values"] = all_q
        save_kwargs["mcts_policy_priors"] = all_priors
        save_kwargs["mcts_temperatures"] = all_temps
        save_kwargs["mcts_root_values"] = all_root_values
        save_kwargs["is_teacher"] = all_is_teacher
        save_kwargs["mcts_first_eval_values"] = all_first_eval
        save_kwargs["mcts_max_subtree_depths"] = all_max_depth

    np.savez_compressed(filepath, **save_kwargs)
    return filepath
