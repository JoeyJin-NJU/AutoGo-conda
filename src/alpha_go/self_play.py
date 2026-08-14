"""Self-play module for generating game data between agents."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from alpha_go.agents import Agent, get_agent, list_agents
from alpha_go.gameplay import (
    play_game,
    save_game_data,
    BLACK,
    WHITE,
)

GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()

# Per-agent shared inference engines for batched mode
# Maps agent_name -> LocalBatchedInferenceEngine
_shared_inference_engines: dict[str, object] = {}

# Map agent names to their model configs
_AGENT_MODEL_CONFIGS: dict[str, str] = {
    "cpp-mcts-3m-v0": "3M",
    "cpp-mcts-3m-explore": "3M",
    "cpp-mcts-18m": "18M",
    "cpp-mcts-18m-v2": "18M",
    "cpp-mcts-18m-v3": "18M",
    "cpp-mcts-18m-v2-lb": "18M",
    "cpp-mcts-18m-v3-lb": "18M",
}


def _detect_model_config(agent_name: str) -> str:
    """Detect model config key for a cpp-mcts agent."""
    if agent_name in _AGENT_MODEL_CONFIGS:
        return _AGENT_MODEL_CONFIGS[agent_name]
    raise ValueError(f"Unknown model config for agent {agent_name}. Add to _AGENT_MODEL_CONFIGS.")

console = Console()


@dataclass
class GameResult:
    """Lightweight result for parallel execution (no large arrays)."""

    winner: int | None
    result: str
    num_moves: int
    game_index: int
    filepath: str | None = None
    termination: str = ""  # "double_pass" or "max_moves"
    black_move_seconds: float = 0.0
    white_move_seconds: float = 0.0
    black_move_count: int = 0
    white_move_count: int = 0


def parse_score_from_result(result: str) -> float:
    """Parse numeric score from result string.

    Convention: positive = Black wins, negative = White wins.
    E.g., "B+2.5" -> 2.5, "W+3.0" -> -3.0, "Draw" -> 0.0
    """
    if result == "Draw" or not result:
        return 0.0
    if result.startswith("B+"):
        return float(result[2:])
    if result.startswith("W+"):
        return -float(result[2:])
    return 0.0


def compute_stats(results: list[GameResult]) -> dict:
    """Compute statistics from game results."""
    total = len(results)
    black_wins = sum(1 for r in results if r.winner == BLACK)
    white_wins = sum(1 for r in results if r.winner == WHITE)
    draws = total - black_wins - white_wins

    move_counts = [r.num_moves for r in results]
    scores = [parse_score_from_result(r.result) for r in results]

    # Count termination types
    double_pass_count = sum(1 for r in results if r.termination == "double_pass")
    max_moves_count = sum(1 for r in results if r.termination == "max_moves")
    resign_count = sum(1 for r in results if r.termination == "resign")

    return {
        "total_games": total,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_win_rate": black_wins / total if total > 0 else 0.0,
        "white_win_rate": white_wins / total if total > 0 else 0.0,
        "avg_game_length": sum(move_counts) / total if total > 0 else 0.0,
        "min_game_length": min(move_counts) if move_counts else 0,
        "max_game_length": max(move_counts) if move_counts else 0,
        "total_positions": sum(move_counts),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_std": float(np.std(scores)) if scores else 0.0,
        "double_pass_ends": double_pass_count,
        "max_moves_ends": max_moves_count,
        "resign_ends": resign_count,
        "black_sec_per_move": (
            sum(r.black_move_seconds for r in results)
            / max(1, sum(r.black_move_count for r in results))
        ),
        "white_sec_per_move": (
            sum(r.white_move_seconds for r in results)
            / max(1, sum(r.white_move_count for r in results))
        ),
        "black_total_seconds": sum(r.black_move_seconds for r in results),
        "white_total_seconds": sum(r.white_move_seconds for r in results),
    }


def compute_stats_from_npz_files(npz_files: list[Path]) -> dict:
    """Compute statistics from npz files."""
    results = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        winner_val = data["winner"].item()
        winner = winner_val if winner_val != 0 else None
        # Handle older files that don't have termination field
        termination = str(data["termination"]) if "termination" in data else ""
        results.append(GameResult(
            winner=winner,
            result=str(data["result"]),
            num_moves=int(data["num_moves"]),
            game_index=0,
            filepath=str(npz_file),
            termination=termination,
        ))
    return compute_stats(results)


# Thread-local storage for agents (avoids recreating agents per game in threaded mode)
_thread_local = threading.local()


def _get_or_create_agent(agent_name: str, role: str) -> Agent:
    """Get or create an agent by name and role using thread-local caching.

    Keyed by (agent_name, role) so that black and white get separate instances
    even when using the same agent type — they have independent engine state.

    When a shared inference engine is available, cpp-mcts agents will use it
    for batched inference across all threads.
    """
    if not hasattr(_thread_local, "agents"):
        _thread_local.agents = {}

    key = f"{agent_name}:{role}"
    if key not in _thread_local.agents:
        if agent_name in _shared_inference_engines:
            from alpha_go.agents.nn_mcts import create_batched_cpp_mcts_agent, CppMCTSAgent
            ref_agent = get_agent(agent_name)
            assert isinstance(ref_agent, CppMCTSAgent)
            ckpt = ref_agent.checkpoint_path
            assert ckpt is not None
            agent = create_batched_cpp_mcts_agent(
                engine=_shared_inference_engines[agent_name],
                checkpoint_path=ckpt,
                board_size=ref_agent.board_size,
                num_simulations=ref_agent.num_simulations,
                c_puct=ref_agent.cpp_config.c_puct,
                temperature=ref_agent.cpp_config.temperature,
                lambda_=ref_agent.cpp_config.lambda_,
                max_depth=ref_agent.cpp_config.max_depth,
            )
            ref_agent.close()
            _thread_local.agents[key] = agent
        else:
            _thread_local.agents[key] = get_agent(agent_name)

    return _thread_local.agents[key]


def _cleanup_thread_local_agents() -> None:
    """Clean up thread-local agents (call from main thread after pool completes)."""
    if hasattr(_thread_local, "agents"):
        for agent in _thread_local.agents.values():
            if hasattr(agent, "close"):
                agent.close()
        _thread_local.agents = {}


class MemoryProfiler:
    """Background thread that logs memory usage to a TSV file."""

    def __init__(self, path: Path, interval: float = 2.0):
        self.path = path
        self.interval = interval
        self.episodes = 0
        self.total_steps = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.t0 = time.monotonic()
        with open(self.path, "w") as f:
            f.write("wall_s\tepisodes\ttotal_steps\trss_mb\tvms_mb\tnum_fds\tnum_children\tfd_types\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def record_episode(self, num_moves: int) -> None:
        with self._lock:
            self.episodes += 1
            self.total_steps += num_moves

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        pid = os.getpid()
        while not self._stop.wait(self.interval):
            try:
                # Read /proc for memory stats (no psutil dependency)
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
                rss_kb = vms_kb = 0
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                    elif line.startswith("VmSize:"):
                        vms_kb = int(line.split()[1])
                # Count open file descriptors
                num_fds = len(os.listdir(f"/proc/{pid}/fd"))
                # Count child processes
                with open(f"/proc/{pid}/status") as f:
                    pass  # already read
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                num_children = len([c for c in children if c.strip()])
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue

            with self._lock:
                eps = self.episodes
                steps = self.total_steps

            # Snapshot FD targets to detect what's leaking
            fd_types: dict[str, int] = {}
            fd_dir = f"/proc/{pid}/fd"
            for fd_name in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd_name}")
                    # Bucket by type: pipe, socket, file, anon_inode, etc.
                    if target.startswith("pipe:"):
                        key = "pipe"
                    elif target.startswith("socket:"):
                        key = "socket"
                    elif target.startswith("anon_inode:"):
                        key = "anon_inode"
                    elif target.startswith("/"):
                        key = "file"
                    else:
                        key = "other"
                    fd_types[key] = fd_types.get(key, 0) + 1
                except OSError:
                    pass
            fd_summary = ";".join(f"{k}={v}" for k, v in sorted(fd_types.items()))

            wall_s = time.monotonic() - self.t0
            with open(self.path, "a") as f:
                f.write(f"{wall_s:.1f}\t{eps}\t{steps}\t{rss_kb / 1024:.1f}\t{vms_kb / 1024:.1f}\t{num_fds}\t{num_children}\t{fd_summary}\n")


_profiler: MemoryProfiler | None = None

# Worker function for parallel execution using threading
def _play_game_worker(work_item: tuple) -> GameResult:  # type: ignore[type-arg]
    """Worker function to play a single game in a thread."""
    (
        game_index,
        cli_args,
        output_dir_str,
        date_slug,
        save_data,
        game_index_offset,
        max_moves,
        komi,
    ) = work_item

    # Apply offset to get the actual game index for file naming
    actual_game_index = game_index + game_index_offset

    # Seed random state uniquely for this game
    game_seed = actual_game_index + cli_args.seed

    try:
        # Get or create agents using thread-local caching
        black_agent = _get_or_create_agent(cli_args.black, "black")
        white_agent = _get_or_create_agent(cli_args.white, "white")

        # Play game
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=cli_args.board_size,
            seed=game_seed,
            max_moves=max_moves,
            komi=komi,
            collect_metrics=getattr(cli_args, "collect_metrics", False),
            black_is_teacher=getattr(cli_args, "black_is_teacher", False),
            white_is_teacher=getattr(cli_args, "white_is_teacher", False),
        )
    except Exception:
        import traceback
        print(f"[FATAL] Game {actual_game_index} crashed:\n{traceback.format_exc()}", flush=True)
        raise

    if _profiler:
        _profiler.record_episode(record.num_moves)

    # Save if requested
    filepath = None
    if save_data and output_dir_str:
        output_dir = Path(output_dir_str)
        filepath = str(save_game_data(record, output_dir, actual_game_index, date_slug))

    return GameResult(
        winner=record.winner,
        result=record.result,
        num_moves=record.num_moves,
        game_index=actual_game_index,
        filepath=filepath,
        termination=record.termination,
        black_move_seconds=record.black_move_seconds,
        white_move_seconds=record.white_move_seconds,
        black_move_count=record.black_move_count,
        white_move_count=record.white_move_count,
    )


def run_sequential(
    args,
    output_dir: Path | None,
    date_slug: str,
) -> list[GameResult]:
    """Run games sequentially with live progress."""
    black_agent = _get_or_create_agent(args.black, "black")
    white_agent = _get_or_create_agent(args.white, "white")
    offset = getattr(args, 'game_index_offset', 0)

    results: list[GameResult] = []
    black_wins = 0
    white_wins = 0

    for i in range(args.num_games):
        actual_idx = i + offset
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=args.board_size,
            seed=actual_idx + args.seed,
            max_moves=args.max_moves,
            komi=args.komi,
            collect_metrics=getattr(args, "collect_metrics", False),
            black_is_teacher=getattr(args, "black_is_teacher", False),
            white_is_teacher=getattr(args, "white_is_teacher", False),
        )

        if _profiler:
            _profiler.record_episode(record.num_moves)

        # Save immediately after each game
        filepath = None
        if args.save_name and output_dir:
            filepath = str(save_game_data(record, output_dir, actual_idx, date_slug))

        result = GameResult(
            winner=record.winner,
            result=record.result,
            num_moves=record.num_moves,
            game_index=actual_idx,
            filepath=filepath,
            termination=record.termination,
            black_move_seconds=record.black_move_seconds,
            white_move_seconds=record.white_move_seconds,
            black_move_count=record.black_move_count,
            white_move_count=record.white_move_count,
        )
        results.append(result)

        if record.winner == BLACK:
            black_wins += 1
        elif record.winner == WHITE:
            white_wins += 1

        total = i + 1
        black_rate = black_wins / total
        white_rate = white_wins / total
        b_spm = record.black_move_seconds / max(1, record.black_move_count)
        w_spm = record.white_move_seconds / max(1, record.white_move_count)
        console.print(
            f"[bold]Game {total}/{args.num_games}[/bold]: {record.result} | "
            f"[black on white] {args.black} [/black on white] "
            f"[green]{black_wins}/{total}[/green] ({black_rate:.1%}) "
            f"[dim]{b_spm:.3f}s/mv[/dim] - "
            f"[green]{white_wins}/{total}[/green] ({white_rate:.1%}) "
            f"[dim]{w_spm:.3f}s/mv[/dim] "
            f"[white on black] {args.white} [/white on black]"
        )

    return results


def run_parallel(
    args,
    output_dir: Path | None,
    date_slug: str,
) -> list[GameResult]:
    """Run games in parallel using threading."""
    save_data = args.save_name is not None
    output_dir_str = str(output_dir) if output_dir else None
    offset = getattr(args, 'game_index_offset', 0)

    # Prepare work items
    work_items = [
        (
            i,
            args,
            output_dir_str,
            date_slug,
            save_data,
            offset,
            args.max_moves,
            args.komi,
        )
        for i in range(args.num_games)
    ]

    results: list[GameResult] = []
    errors: list[str] = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_play_game_worker, item): i for i, item in enumerate(work_items)}

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                errors.append(str(e))
                if len(errors) <= 3:
                    print(f"[ERROR] Worker failed: {e}", flush=True)
                if len(errors) >= args.num_workers:
                    print(f"[FATAL] All workers failed ({len(errors)} errors), aborting.", flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                continue

            results.append(result)
            total = len(results)
            black_wins = sum(1 for r in results if r.winner == BLACK)
            white_wins = sum(1 for r in results if r.winner == WHITE)
            elapsed = time.monotonic() - t0
            games_per_min = total / elapsed * 60 if elapsed > 0 else 0

            # Print every game so Ray logs capture progress
            b_spm = result.black_move_seconds / max(1, result.black_move_count)
            w_spm = result.white_move_seconds / max(1, result.white_move_count)
            print(
                f"Game {total}/{args.num_games}: {result.result} ({result.num_moves} moves, {result.termination}) | "
                f"{args.black} {black_wins} ({b_spm:.3f}s/mv) - "
                f"{white_wins} ({w_spm:.3f}s/mv) {args.white} "
                f"({games_per_min:.1f} games/min)",
                flush=True,
            )

    if errors:
        print(f"[WARN] {len(errors)} games failed out of {args.num_games}", flush=True)

    # Sort by game index for consistent ordering
    results.sort(key=lambda r: r.game_index)
    return results


def main() -> None:
    available_agents = list_agents()
    agents_help = f"Available: {', '.join(available_agents)}"

    parser = argparse.ArgumentParser(description="Self-play between Go agents")
    parser.add_argument(
        "--black",
        type=str,
        default="gnugo1",
        help=f"Agent for black. {agents_help}",
    )
    parser.add_argument(
        "--white",
        type=str,
        default="gnugo1",
        help=f"Agent for white. {agents_help}",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10,
        help="Number of games to play",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=9,
        choices=[9, 13, 19],
        help="Board size",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Max number of moves played before final area score is calculated. "
             "Defaults to board_size * board_size * 2."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
        help="Komi compensation for white (default: 7.5)",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="slug name for output data. Data not written if not set.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential)",
    )
    parser.add_argument(
        "--game_index_offset",
        type=int,
        default=0,
        help="Offset for game indices (used by distributed execution)",
    )
    parser.add_argument(
        "--profile-memory",
        type=str,
        default=None,
        metavar="PATH",
        help="Write memory profile TSV to this path (sampled every 2s)",
    )
    parser.add_argument(
        "--batched-inference",
        action="store_true",
        help="Use shared batched inference engine for cpp-mcts agents (higher GPU utilization)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batched inference engine (default: 64)",
    )
    parser.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=1.0,
        help="Batch collection timeout in ms (default: 1.0)",
    )
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        default=False,
        help="Save MCTS visit counts and search stats in NPZ files",
    )
    parser.add_argument(
        "--black-is-teacher",
        action="store_true",
        default=False,
        help="Tag black-side MoveMetrics with is_teacher=True (saved in NPZ)",
    )
    parser.add_argument(
        "--white-is-teacher",
        action="store_true",
        default=False,
        help="Tag white-side MoveMetrics with is_teacher=True (saved in NPZ)",
    )
    args = parser.parse_args()

    # Default max_moves scales with board size: a Go game's natural ceiling is
    # ~2 * board_size**2 moves (every point played twice — once placed, once
    # recaptured). Avoids hardcoded constants like 300 that cap 19x19 games
    # below the natural game length.
    if args.max_moves is None:
        args.max_moves = args.board_size * args.board_size * 2

    # Local execution mode
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.save_name:
        output_dir = GAME_DATA_DIR / args.save_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    date_slug = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set up shared batched inference engines if requested (one per unique cpp-mcts agent)
    global _shared_inference_engines
    if args.batched_inference:
        from alpha_go.agents.nn_mcts import create_shared_inference_engine, CppMCTSAgent, LocalNNEvaluator
        for agent_name in [args.black, args.white]:
            if agent_name.startswith("cpp-mcts-") and agent_name not in _shared_inference_engines:
                ref = get_agent(agent_name)
                assert isinstance(ref, CppMCTSAgent)
                ckpt = ref.checkpoint_path
                board_size = ref.board_size
                evaluator = ref.evaluator
                assert isinstance(evaluator, LocalNNEvaluator)
                model_config = _detect_model_config(agent_name)
                ref.close()
                assert ckpt is not None
                _shared_inference_engines[agent_name] = create_shared_inference_engine(
                    checkpoint_path=ckpt,
                    model_config=model_config,
                    board_size=board_size,
                    batch_size=args.batch_size,
                    batch_timeout_ms=args.batch_timeout_ms,
                )
                console.print(
                    f"[green]Batched inference engine for {agent_name}: "
                    f"model={model_config}, batch_size={args.batch_size}, "
                    f"timeout={args.batch_timeout_ms}ms[/green]"
                )

    # Start memory profiler if requested
    global _profiler
    if args.profile_memory:
        profile_path = Path(args.profile_memory)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        _profiler = MemoryProfiler(profile_path)
        _profiler.start()
        console.print(f"[yellow]Memory profiler writing to {profile_path}[/yellow]")

    # Run games with offset for game indices
    if args.num_workers > 1:
        results = run_parallel(args, output_dir, date_slug)
    else:
        results = run_sequential(args, output_dir, date_slug)

    if _profiler:
        _profiler.stop()
        console.print(f"[yellow]Memory profile saved to {args.profile_memory}[/yellow]")

    # Stop shared inference engines
    for name, engine in _shared_inference_engines.items():
        stats = engine.get_stats()
        console.print(f"[green]{name} inference stats: {stats}[/green]")
        engine.stop()
    _shared_inference_engines.clear()

    # Compute stats
    stats = compute_stats(results)
    stats["black_agent"] = args.black
    stats["white_agent"] = args.white
    stats["board_size"] = args.board_size
    stats["num_workers"] = args.num_workers
    if output_dir:
        stats["data_files"] = [r.filepath for r in results if r.filepath]
        stats["output_dir"] = str(output_dir)

    # Print summary table
    table = Table(title="Self-Play Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Games", str(stats["total_games"]))

    black_row = f"{stats['black_wins']}/{stats['total_games']} ({stats['black_win_rate']:.1%})"
    table.add_row(f"{args.black} (Black) Wins", black_row)

    white_row = f"{stats['white_wins']}/{stats['total_games']} ({stats['white_win_rate']:.1%})"
    table.add_row(f"{args.white} (White) Wins", white_row)

    table.add_row("Draws", str(stats["draws"]))
    table.add_row("Score (Black perspective)", f"{stats['score_mean']:.1f} ± {stats['score_std']:.1f}")
    table.add_row("Avg Game Length", f"{stats['avg_game_length']:.1f}")
    table.add_row(f"{args.black} (Black) sec/move", f"{stats['black_sec_per_move']:.3f}")
    table.add_row(f"{args.white} (White) sec/move", f"{stats['white_sec_per_move']:.3f}")
    # Show termination breakdown
    double_pass = stats.get("double_pass_ends", 0)
    max_moves = stats.get("max_moves_ends", 0)
    table.add_row("Ended by Double Pass", f"{double_pass}/{stats['total_games']}")
    table.add_row("Ended by Max Moves", f"{max_moves}/{stats['total_games']}")
    if args.num_workers > 1:
        table.add_row("Workers", str(args.num_workers))
    if stats.get("output_dir"):
        table.add_row("Output Dir", stats["output_dir"])
    console.print(table)

    # Also print JSON for programmatic use
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
