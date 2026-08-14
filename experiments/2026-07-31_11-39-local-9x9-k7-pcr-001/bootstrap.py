#!/usr/bin/env python
"""Bounded, process-parallel random-vs-random bootstrap collector."""
from __future__ import annotations

import os

# Every worker is already an independent process. Prevent imported numerical
# libraries from creating nested thread pools on top of the process pool.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from common import EXP_NAME, GAME_DATA_ROOT, load_config, validate_npz


GAME_INDEX_RE = re.compile(r"-game(?P<index>\d{7})\.npz$")
_black_agent: Any = None
_white_agent: Any = None


def extract_game_index(path: Path) -> int:
    """Read the stable numeric game index from an AutoGo NPZ filename."""
    match = GAME_INDEX_RE.search(path.name)
    if match is None:
        raise ValueError(f"unrecognized bootstrap filename: {path.name}")
    return int(match.group("index"))


def discover_game_indices(output_dir: Path) -> dict[int, Path]:
    """Return published files keyed by game index, rejecting duplicates."""
    indexed: dict[int, Path] = {}
    for path in sorted(output_dir.glob("*.npz")):
        index = extract_game_index(path)
        if index in indexed:
            raise ValueError(
                f"duplicate bootstrap game index {index}: "
                f"{indexed[index].name}, {path.name}"
            )
        indexed[index] = path
    return indexed


def _init_worker() -> None:
    """Create one pair of reusable random agents per process."""
    global _black_agent, _white_agent
    from alpha_go.agents import get_agent

    _black_agent = get_agent("random")
    _white_agent = get_agent("random")


def _play_validate_publish(task: tuple[Any, ...]) -> dict[str, Any]:
    """Generate one game and publish it only after NPZ validation."""
    (
        game_index,
        game_seed,
        board_size,
        komi,
        max_moves,
        output_dir_str,
        date_slug,
    ) = task

    from alpha_go.gameplay import play_game, save_game_data

    output_dir = Path(output_dir_str)
    worker_dir = output_dir / ".bootstrap-process-tmp" / f"pid-{os.getpid()}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    record = play_game(
        black_agent=_black_agent,
        white_agent=_white_agent,
        board_size=board_size,
        seed=game_seed,
        max_moves=max_moves,
        komi=komi,
        collect_metrics=False,
        black_is_teacher=False,
        white_is_teacher=False,
    )
    temporary = save_game_data(record, worker_dir, game_index, date_slug)
    validate_npz(temporary, board_size=board_size, require_mcts=False)

    destination = output_dir / temporary.name
    os.replace(temporary, destination)
    return {
        "game_index": game_index,
        "winner": int(record.winner or 0),
        "num_moves": int(record.num_moves),
        "termination": record.termination,
        "path": str(destination),
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=cfg["bootstrap"]["games"])
    parser.add_argument("--num-workers", type=int, default=cfg["bootstrap"]["workers"])
    parser.add_argument("--save-name", required=True)
    parser.add_argument("--seed", type=int, default=cfg["base_seed"])
    parser.add_argument("--game-index-offset", type=int, default=0)
    parser.add_argument("--max-moves", type=int, default=cfg["rules"]["max_moves"])
    args = parser.parse_args()

    if args.num_games < 0:
        raise ValueError("--num-games must be non-negative")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")

    output_dir = (GAME_DATA_ROOT / args.save_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_total = args.game_index_offset + args.num_games
    published = discover_game_indices(output_dir)
    unexpected = sorted(index for index in published if index >= target_total)
    if unexpected:
        raise ValueError(
            f"published bootstrap indices exceed target {target_total}: "
            f"{unexpected[:10]}"
        )
    missing = [index for index in range(target_total) if index not in published]
    workers = min(args.num_workers, len(missing)) if missing else 0
    date_slug = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(
        f"Bootstrap {EXP_NAME}: target={target_total} existing={len(published)} "
        f"remaining={len(missing)} processes={workers} "
        f"board={cfg['rules']['board_size']} komi={cfg['rules']['komi']} "
        f"output={args.save_name}",
        flush=True,
    )
    if not missing:
        return

    tasks = [
        (
            index,
            args.seed + index,
            int(cfg["rules"]["board_size"]),
            float(cfg["rules"]["komi"]),
            args.max_moves,
            str(output_dir),
            date_slug,
        )
        for index in missing
    ]
    completed = 0
    black_wins = 0
    white_wins = 0
    t0 = time.monotonic()
    progress_interval = max(1, min(25, len(tasks) // 20 or 1))

    executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
    futures = {executor.submit(_play_validate_publish, task): task[0] for task in tasks}
    try:
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result["winner"] == 1:
                black_wins += 1
            elif result["winner"] == 2:
                white_wins += 1
            if completed == 1 or completed % progress_interval == 0 or completed == len(tasks):
                elapsed = time.monotonic() - t0
                rate = completed / elapsed * 60 if elapsed else 0.0
                print(
                    f"Bootstrap progress: {completed}/{len(tasks)} new, "
                    f"published={len(published) + completed}/{target_total}, "
                    f"B/W={black_wins}/{white_wins}, {rate:.1f} games/min",
                    flush=True,
                )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    final = discover_game_indices(output_dir)
    expected = set(range(target_total))
    if set(final) != expected:
        missing_after = sorted(expected - set(final))
        raise RuntimeError(
            f"bootstrap publication incomplete: {len(final)}/{target_total}; "
            f"first missing indices={missing_after[:10]}"
        )


if __name__ == "__main__":
    main()
