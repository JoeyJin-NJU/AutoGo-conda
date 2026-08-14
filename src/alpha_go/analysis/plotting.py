"""Plotting utilities for training analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Go board constants
BLACK = 1
WHITE = 2


def load_training_run(run_name: str, data_dir: str = "research_reports/data") -> tuple[pd.DataFrame, dict]:
    """Load training CSV and flops JSON for a run."""
    data_path = Path(data_dir)
    csv_path = data_path / f"{run_name}.csv"
    flops_path = data_path / f"{run_name}_flops.json"

    df = pd.read_csv(csv_path)
    with open(flops_path) as f:
        flops_info = json.load(f)

    return df, flops_info


def compute_cumulative_flops(df: pd.DataFrame, flops_info: dict) -> pd.Series:
    """Compute cumulative training FLOPs using 6*N*T per sample.

    Args:
        df: Training dataframe with global_step column
        flops_info: Dict containing n_params and board_size

    Returns:
        Series of cumulative FLOPs at each step
    """
    n_params = flops_info["n_params"]
    board_size = flops_info["board_size"]
    n_tokens = board_size ** 2 + 1  # board positions + pass token
    flops_per_sample = 6 * n_params * n_tokens

    # Assume batch_size from the data (infer from steps per epoch)
    # Each global_step processes one batch
    # For now, use global_step * batch_size * flops_per_sample
    # We'll estimate batch_size from the CSV metadata if available
    batch_size = df.get("batch_size", pd.Series([64] * len(df))).iloc[0]
    if pd.isna(batch_size):
        batch_size = 64  # default

    return df["global_step"] * batch_size * flops_per_sample


def plot_loss_vs_flops(
    run_name: str,
    data_dir: str = "research_reports/data",
    output_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot training and validation loss as a function of cumulative FLOPs.

    Args:
        run_name: Name of the training run (without extension)
        data_dir: Directory containing CSV and JSON files
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    df, flops_info = load_training_run(run_name, data_dir)
    cumulative_flops = compute_cumulative_flops(df, flops_info)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot training loss (per-batch, more dense)
    train_mask = df["train_loss"].notna()
    ax.plot(
        cumulative_flops[train_mask],
        df.loc[train_mask, "train_loss"],
        alpha=0.3,
        color="blue",
        label="Train loss (batch)",
        linewidth=0.5,
    )

    # Plot validation loss (per-epoch, sparse)
    val_mask = df["val_loss"].notna()
    ax.plot(
        cumulative_flops[val_mask],
        df.loc[val_mask, "val_loss"],
        color="orange",
        marker="o",
        markersize=6,
        label="Val loss",
        linewidth=2,
    )

    # Plot train eval loss (per-epoch, sparse)
    train_eval_mask = df["train_eval_loss"].notna()
    ax.plot(
        cumulative_flops[train_eval_mask],
        df.loc[train_eval_mask, "train_eval_loss"],
        color="blue",
        marker="s",
        markersize=6,
        label="Train loss (epoch)",
        linewidth=2,
    )

    ax.set_xlabel("Cumulative Training FLOPs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss vs Training FLOPs\n{run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("linear")

    # Format x-axis with scientific notation
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_vs_flops(
    run_name: str,
    data_dir: str = "research_reports/data",
    output_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot policy and value accuracy as a function of cumulative FLOPs.

    Args:
        run_name: Name of the training run (without extension)
        data_dir: Directory containing CSV and JSON files
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    df, flops_info = load_training_run(run_name, data_dir)
    cumulative_flops = compute_cumulative_flops(df, flops_info)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Policy accuracy
    mask = df["val_policy_acc"].notna()
    ax1.plot(
        cumulative_flops[mask],
        df.loc[mask, "train_policy_acc"] * 100,
        color="blue",
        marker="s",
        label="Train",
    )
    ax1.plot(
        cumulative_flops[mask],
        df.loc[mask, "val_policy_acc"] * 100,
        color="orange",
        marker="o",
        label="Val",
    )
    ax1.set_xlabel("Cumulative Training FLOPs")
    ax1.set_ylabel("Policy Accuracy (%)")
    ax1.set_title("Policy Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    # Value accuracy
    ax2.plot(
        cumulative_flops[mask],
        df.loc[mask, "train_value_acc"] * 100,
        color="blue",
        marker="s",
        label="Train",
    )
    ax2.plot(
        cumulative_flops[mask],
        df.loc[mask, "val_value_acc"] * 100,
        color="orange",
        marker="o",
        label="Val",
    )
    ax2.set_xlabel("Cumulative Training FLOPs")
    ax2.set_ylabel("Value Accuracy (%)")
    ax2.set_title("Value Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    plt.suptitle(f"Accuracy vs Training FLOPs\n{run_name}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_summary(
    run_name: str,
    data_dir: str = "research_reports/data",
    output_dir: str = "research_reports/figures",
) -> list[Path]:
    """Generate and save all training plots for a run.

    Args:
        run_name: Name of the training run
        data_dir: Directory containing CSV and JSON files
        output_dir: Directory to save figures

    Returns:
        List of paths to saved figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # Loss plot
    loss_path = output_path / f"{run_name}_loss.png"
    plot_loss_vs_flops(run_name, data_dir, str(loss_path))
    saved_paths.append(loss_path)
    plt.close()

    # Accuracy plot
    acc_path = output_path / f"{run_name}_accuracy.png"
    plot_accuracy_vs_flops(run_name, data_dir, str(acc_path))
    saved_paths.append(acc_path)
    plt.close()

    return saved_paths


# =============================================================================
# Go Board Plotting
# =============================================================================


def get_star_points(board_size: int) -> list[tuple[int, int]]:
    """Get star point coordinates for a given board size."""
    if board_size == 9:
        return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    elif board_size == 13:
        return [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
    elif board_size == 19:
        return [
            (3, 3), (3, 9), (3, 15),
            (9, 3), (9, 9), (9, 15),
            (15, 3), (15, 9), (15, 15),
        ]
    return []


def get_column_labels(board_size: int) -> list[str]:
    """Get column labels for a Go board (skipping 'I')."""
    letters = "ABCDEFGHJKLMNOPQRST"  # Standard Go notation skips 'I'
    return list(letters[:board_size])


def render_board_simple(
    board: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    board_size: int = 9,
) -> None:
    """Render a simple Go board without policy overlay.

    Args:
        board: 2D numpy array with 0=empty, 1=black, 2=white
        ax: Matplotlib axes to draw on
        title: Title for the plot
        board_size: Size of the board (default 9)
    """
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(board_size - 0.5, -0.5)
    ax.set_aspect("equal")

    # Draw grid
    for i in range(board_size):
        ax.axhline(i, color="#DEB887", linewidth=0.5)
        ax.axvline(i, color="#DEB887", linewidth=0.5)

    # Star points
    for row, col in get_star_points(board_size):
        ax.plot(col, row, "ko", markersize=3)

    # Draw stones
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] == BLACK:
                circle = plt.Circle(
                    (col, row), 0.4, facecolor="black", edgecolor="black"
                )
                ax.add_patch(circle)
            elif board[row, col] == WHITE:
                circle = plt.Circle(
                    (col, row), 0.4, facecolor="white", edgecolor="black"
                )
                ax.add_patch(circle)

    # Labels
    ax.set_xticks(range(board_size))
    ax.set_xticklabels(get_column_labels(board_size))
    ax.set_yticks(range(board_size))
    ax.set_yticklabels([str(board_size - i) for i in range(board_size)])
    ax.set_facecolor("#DEB887")
    ax.set_title(title)


def render_board_with_policy(
    board: np.ndarray | list[list[int]],
    policy_2d: np.ndarray,
    ax: plt.Axes,
    legal_moves: list[tuple[int, int]] | None = None,
    chosen_move: tuple[int, int] | None = None,
    title: str = "",
    board_size: int = 9,
    show_probs: bool = True,
    prob_threshold: float = 0.05,
) -> None:
    """Render Go board with policy heatmap overlay.

    Args:
        board: 2D array with 0=empty, 1=black, 2=white
        policy_2d: 2D array of policy probabilities (same shape as board)
        ax: Matplotlib axes to draw on
        legal_moves: List of legal move coordinates. If None, all non-stone positions shown.
        chosen_move: Coordinates of the chosen/actual move to highlight
        title: Title for the plot
        board_size: Size of the board (default 9)
        show_probs: Whether to show probability text on high-prob moves
        prob_threshold: Minimum probability to show text label
    """
    # Convert board to numpy if needed
    if isinstance(board, list):
        board = np.array(board)

    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(board_size - 0.5, -0.5)
    ax.set_aspect("equal")

    # Draw grid
    for i in range(board_size):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)

    # Star points
    for row, col in get_star_points(board_size):
        ax.plot(col, row, "ko", markersize=4)

    # Determine which squares to show policy for
    if legal_moves is not None:
        legal_set = set(legal_moves)
    else:
        # Show policy for all empty squares
        legal_set = {
            (r, c)
            for r in range(board_size)
            for c in range(board_size)
            if board[r, c] == 0
        }

    # Draw policy heatmap for legal moves only
    for row in range(board_size):
        for col in range(board_size):
            if (row, col) in legal_set and policy_2d[row, col] > 0.001:
                alpha = min(policy_2d[row, col] * 3, 0.8)
                rect = plt.Rectangle(
                    (col - 0.4, row - 0.4),
                    0.8,
                    0.8,
                    facecolor="green",
                    alpha=alpha,
                    edgecolor="none",
                )
                ax.add_patch(rect)
                # Show probability text for high-probability moves
                if show_probs and policy_2d[row, col] > prob_threshold:
                    ax.text(
                        col,
                        row,
                        f"{policy_2d[row, col]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="darkgreen",
                    )

    # Draw stones
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] == BLACK:
                circle = plt.Circle(
                    (col, row), 0.4, facecolor="black", edgecolor="black"
                )
                ax.add_patch(circle)
            elif board[row, col] == WHITE:
                circle = plt.Circle(
                    (col, row), 0.4, facecolor="white", edgecolor="black"
                )
                ax.add_patch(circle)

    # Highlight chosen move
    if chosen_move is not None:
        row, col = chosen_move
        ax.plot(col, row, "rx", markersize=12, markeredgewidth=3)

    # Labels
    ax.set_xticks(range(board_size))
    ax.set_xticklabels(get_column_labels(board_size))
    ax.set_yticks(range(board_size))
    ax.set_yticklabels([str(board_size - i) for i in range(board_size)])
    ax.set_title(title)
