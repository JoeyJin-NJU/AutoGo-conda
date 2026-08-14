#!/usr/bin/env python
"""Run one bounded MCTS collection or arena shard on one visible GPU."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

import alpha_go_cpp
import numpy as np
import torch
from color_model import SignedKomiLeafBatchedNNEvaluator
from common import GAME_DATA_ROOT, atomic_write_json, load_config, utc_timestamp

from alpha_go.agents.base import PASS, RESIGN, register_agent
from alpha_go.agents.nn_mcts import CppMCTSAgent


def _tag(path: str) -> str:
    match = re.search(r"iter(\d+)", path)
    stem = match.group(1) if match else os.path.basename(path).replace(".pt", "")
    return re.sub(r"[^A-Za-z0-9_-]", "", stem)[-32:]


def choose_pcr_budget(seed: int, simulations: list[int], probabilities: list[float]) -> int:
    """Sample one deterministic per-move playout cap from the configured PCR mix."""
    return int(np.random.default_rng(seed).choice(simulations, p=probabilities))


def chosen_move_temperature(
    turn_number: int,
    board_size: int,
    *,
    early: float,
    halflife: float,
    late: float,
) -> float:
    """Match KataGo's board-size-scaled exponential move-temperature decay."""
    if turn_number < 0:
        raise ValueError("turn number must be non-negative")
    if board_size <= 0:
        raise ValueError("board size must be positive")
    if halflife <= 0.0:
        raise ValueError("temperature halflife must be positive")
    halflives = (float(turn_number) / halflife) * (19.0 / float(board_size))
    return late + (early - late) * math.pow(0.5, halflives)


def _use_root_noise(*, enabled: bool, full_search: bool, full_search_only: bool) -> bool:
    return enabled and (full_search or not full_search_only)


def temperature_schedule_config(
    config: dict[str, object],
    *,
    is_arena: bool,
) -> tuple[float, float, float]:
    """Return mode-specific early, halflife, and late move temperatures."""
    section = config["arena" if is_arena else "mcts"]
    if not isinstance(section, dict):
        raise TypeError("search configuration section must be a mapping")
    base = float(section["action_temperature"])
    return (
        float(section.get("chosen_move_temperature_early", base)),
        float(section.get("chosen_move_temperature_halflife", 19.0)),
        float(section.get("chosen_move_temperature", base)),
    )


def move_search_settings(
    *,
    seed: int,
    turn_number: int,
    board_size: int,
    base_simulations: int,
    pcr_sims: list[int] | None,
    pcr_probs: list[float] | None,
    temperature_early: float,
    temperature_halflife: float,
    temperature_late: float,
    dirichlet_noise: bool,
    dirichlet_full_search_only: bool,
) -> tuple[int, float, bool]:
    """Resolve per-move simulations, sampling temperature, and root noise."""
    if pcr_sims:
        if not pcr_probs:
            raise ValueError("PCR probabilities are required when PCR is enabled")
        simulations = choose_pcr_budget(seed, pcr_sims, pcr_probs)
        full_search = simulations == max(pcr_sims)
    else:
        simulations = int(base_simulations)
        full_search = True
    temperature = chosen_move_temperature(
        turn_number,
        board_size,
        early=temperature_early,
        halflife=temperature_halflife,
        late=temperature_late,
    )
    use_noise = _use_root_noise(
        enabled=dirichlet_noise,
        full_search=full_search,
        full_search_only=dirichlet_full_search_only,
    )
    return simulations, temperature, use_noise


def _mark_full_search_teachers(path: Path, full_simulations: int) -> None:
    """Atomically mark only full-search rows as policy teachers."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    if "mcts_visits" not in payload:
        raise ValueError(f"missing mcts_visits in collected game: {path}")
    totals = np.asarray(payload["mcts_visits"]).sum(axis=1).astype(np.int64)
    payload["is_teacher"] = totals == int(full_simulations)
    tmp = path.with_name(path.name + ".teacher.tmp.npz")
    np.savez_compressed(tmp, **payload)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _rewrite_collected_teacher_flags(
    output_dir: Path,
    *,
    game_index_offset: int,
    num_games: int,
    full_simulations: int,
) -> None:
    for game_index in range(game_index_offset, game_index_offset + num_games):
        matches = sorted(output_dir.glob(f"*-game{game_index:07d}.npz"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one game file for index {game_index}, found {len(matches)}"
            )
        _mark_full_search_teachers(matches[0], full_simulations)


def _game_invocations(
    seed: int,
    game_index_offset: int,
    num_games: int,
) -> list[tuple[int, int]]:
    """Return per-game seed/offset pairs matching separate legacy launches."""
    return [
        ((seed + local_index) % (1 << 32), game_index_offset + local_index)
        for local_index in range(num_games)
    ]


def _reset_reused_search_state(agent: Any) -> None:
    """Clear all per-game Python MCTS state before a cached agent is reused."""
    agent._consec_below = 0
    agent._turns_played = 0
    agent.last_search_result = None


def _make_agent(
    name: str,
    checkpoint: str,
    *,
    board_size: int,
    komi: float,
    model_config: dict[str, object],
    num_simulations: int,
    c_puct: float,
    temperature: float,
    leaf_batch_size: int,
    max_depth: int,
    policy_temperature: float,
    pcr_sims: list[int] | None,
    pcr_probs: list[float] | None,
    resign_threshold: float,
    resign_consecutive_turns: int,
    minimum_own_turns_before_resign: int,
    chosen_move_temperature_early: float,
    chosen_move_temperature_halflife: float,
    chosen_move_temperature_late: float,
    dirichlet_noise: bool,
    dirichlet_full_search_only: bool,
    dirichlet_total_concentration: float,
    dirichlet_weight: float,
) -> str:
    @register_agent(name)
    class _Agent(CppMCTSAgent):
        def __init__(self) -> None:
            evaluator = SignedKomiLeafBatchedNNEvaluator(
                checkpoint_path=checkpoint,
                board_size=board_size,
                komi=komi,
                model_config=model_config,
                policy_temperature=policy_temperature,
            )
            super().__init__(
                evaluator=evaluator,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature=temperature,
                add_noise=False,
                lambda_=0.0,
                rollout_temperature=1.0,
                max_depth=max_depth,
                resign_threshold=resign_threshold,
                resign_consec_turns=resign_consecutive_turns,
                min_turns_before_resign=minimum_own_turns_before_resign,
                leaf_batch_size=leaf_batch_size,
            )
            self._pcr_sims = list(pcr_sims or [])
            self._pcr_probs = list(pcr_probs or [])
            self._chosen_move_temperature_early = chosen_move_temperature_early
            self._chosen_move_temperature_halflife = chosen_move_temperature_halflife
            self._chosen_move_temperature_late = chosen_move_temperature_late
            self._dirichlet_noise = dirichlet_noise
            self._dirichlet_full_search_only = dirichlet_full_search_only
            self._dirichlet_total_concentration = dirichlet_total_concentration
            self._dirichlet_weight = dirichlet_weight

        def start_game(self, requested_board_size: int) -> None:
            if requested_board_size != self.board_size:
                raise ValueError(
                    f"agent board size {self.board_size} != {requested_board_size}"
                )
            _reset_reused_search_state(self)

        def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            simulations, temperature, use_noise = move_search_settings(
                seed=seed,
                turn_number=board.move_count(),
                board_size=board_size,
                base_simulations=num_simulations,
                pcr_sims=self._pcr_sims,
                pcr_probs=self._pcr_probs,
                temperature_early=self._chosen_move_temperature_early,
                temperature_halflife=self._chosen_move_temperature_halflife,
                temperature_late=self._chosen_move_temperature_late,
                dirichlet_noise=self._dirichlet_noise,
                dirichlet_full_search_only=self._dirichlet_full_search_only,
            )
            self.num_simulations = simulations
            self.temperature = temperature
            if use_noise:
                legal_actions = len(board.get_legal_moves_flat()) + 1  # include pass
                self.cpp_config.dirichlet_alpha = (
                    self._dirichlet_total_concentration / max(1, legal_actions)
                )
                self.cpp_config.dirichlet_weight = self._dirichlet_weight
            else:
                self.cpp_config.dirichlet_alpha = 0.0

            result = self.search_from_cpp_board(board)
            self.last_search_result = result
            self._turns_played += 1

            if self.resign_threshold > 0:
                root_q = 1.0 - result.Q
                self._consec_below = (
                    self._consec_below + 1
                    if root_q < self.resign_threshold
                    else 0
                )
                if (
                    self._consec_below >= self.resign_consec_turns
                    and self._turns_played >= self.min_turns_before_resign
                ):
                    return RESIGN

            flat_action = result.tree.select_action(self.temperature)
            move = (
                PASS
                if flat_action == alpha_go_cpp.PASS_ACTION
                else board.row_col(flat_action)
            )
            if self.post_move_callback_fn is not None:
                self.post_move_callback_fn(result)
            return move

    from alpha_go import self_play
    self_play._AGENT_MODEL_CONFIGS[name] = "18M"
    return name


def _run_bounded_games(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    agent_name_cache: dict[tuple[str, str, str], str] | None = None,
) -> None:
    """Run the requested games, optionally reusing registered agents."""
    board_size = int(cfg["rules"]["board_size"])
    mcts = cfg["mcts"]
    arena = cfg["arena"]
    is_arena = args.mode == "arena"
    search = arena if is_arena else mcts
    num_simulations = int(
        args.num_simulations
        if args.num_simulations is not None
        else search["num_simulations"]
    )
    pcr_sims = None if is_arena or args.disable_pcr else list(mcts["pcr_sims"])
    pcr_probs = None if is_arena or args.disable_pcr else list(mcts["pcr_probs"])
    temperature = float(search["action_temperature"])
    temperature_early, temperature_halflife, temperature_late = (
        temperature_schedule_config(cfg, is_arena=is_arena)
    )
    c_puct = float(search["c_puct"])
    leaf_batch_size = int(
        args.leaf_batch_size
        if args.leaf_batch_size is not None
        else search["leaf_batch_size"]
    )
    if leaf_batch_size < 1:
        raise ValueError("leaf batch size must be positive")
    resign_threshold = float(search["resign_threshold"])
    min_resign_turns = int(search.get("minimum_own_turns_before_resign", 0))
    nonce = f"{os.getpid()}-{args.seed}"
    agent_options = {
        "board_size": board_size,
        "komi": float(cfg["rules"]["komi"]),
        "model_config": cfg["model"],
        "num_simulations": num_simulations,
        "c_puct": c_puct,
        "temperature": temperature,
        "leaf_batch_size": leaf_batch_size,
        "max_depth": int(mcts["max_depth"]),
        "policy_temperature": float(mcts["policy_temperature"]),
        "pcr_sims": pcr_sims,
        "pcr_probs": pcr_probs,
        "resign_threshold": resign_threshold,
        "resign_consecutive_turns": int(mcts["resign_consecutive_turns"]),
        "minimum_own_turns_before_resign": min_resign_turns,
        "chosen_move_temperature_early": temperature_early,
        "chosen_move_temperature_halflife": temperature_halflife,
        "chosen_move_temperature_late": temperature_late,
        "dirichlet_noise": bool(search.get("dirichlet_noise", False)),
        "dirichlet_full_search_only": bool(
            search.get("dirichlet_full_search_only", False)
        ),
        "dirichlet_total_concentration": float(
            search.get(
                "dirichlet_total_concentration",
                mcts["dirichlet_total_concentration"],
            )
        ),
        "dirichlet_weight": float(
            search.get("dirichlet_weight", mcts["dirichlet_weight"])
        ),
    }

    def agent_name(role: str, checkpoint: str) -> str:
        key = (args.mode, role, checkpoint)
        if agent_name_cache is None:
            return _make_agent(
                f"{role}-{_tag(checkpoint)}-{nonce}",
                checkpoint,
                **agent_options,
            )
        if key not in agent_name_cache:
            digest = hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()[:10]
            agent_name_cache[key] = _make_agent(
                f"{role}-{_tag(checkpoint)}-{os.getpid()}-{digest}",
                checkpoint,
                **agent_options,
            )
        return agent_name_cache[key]

    black = agent_name("b", args.black_checkpoint)
    white = agent_name("w", args.white_checkpoint)

    from alpha_go.self_play import main as self_play_main
    base_argv = [
        "self_play",
        "--board_size", str(board_size),
        "--komi", str(cfg["rules"]["komi"]),
        "--max-moves", str(args.max_moves or cfg["rules"]["max_moves"]),
        "--num_workers", str(args.num_workers),
        "--save-name", args.save_name,
        "--black", black,
        "--white", white,
    ]
    if not is_arena:
        base_argv.append("--collect-metrics")
    print(
        f"mode={args.mode} games={args.num_games} sims={num_simulations} "
        f"workers={args.num_workers} pcr={pcr_sims} leaf_batch={leaf_batch_size} "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    # Run each game as its own deterministic invocation while keeping the two
    # registered agents alive in this process. This preserves the exact seeds
    # and file indices produced by the legacy one-process-per-game controller,
    # but amortizes Python startup and checkpoint loading across the chunk.
    previous_argv = sys.argv
    try:
        for game_seed, game_index_offset in _game_invocations(
            args.seed, args.game_index_offset, args.num_games
        ):
            sys.argv = [
                *base_argv,
                "--num_games", "1",
                "--seed", str(game_seed),
                "--game_index_offset", str(game_index_offset),
            ]
            self_play_main()
    finally:
        sys.argv = previous_argv
    if not is_arena:
        output_dir = Path(os.environ["GAME_DATA_DIR"]) / args.save_name
        _rewrite_collected_teacher_flags(
            output_dir,
            game_index_offset=args.game_index_offset,
            num_games=args.num_games,
            full_simulations=max(int(value) for value in mcts["pcr_sims"]),
        )


class PersistentGameRunner:
    """Run one dynamically assigned game at a time while caching model agents."""

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        if mode not in ("collect", "arena"):
            raise ValueError(f"unsupported worker mode: {mode}")
        self.config = config
        self.mode = mode
        self.agent_name_cache: dict[tuple[str, str, str], str] = {}

    def run_task(self, task: dict[str, object]) -> dict[str, object]:
        if task.get("command") != "run":
            raise ValueError("persistent worker task command must be 'run'")
        if task.get("mode") != self.mode:
            raise ValueError(
                f"worker mode {self.mode} cannot run task mode {task.get('mode')}"
            )
        log_path = Path(str(task["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        published_save_name = str(task["save_name"])
        task_digest = hashlib.sha256(
            str(task["task_id"]).encode("utf-8")
        ).hexdigest()[:16]
        inflight_save_name = str(
            Path(published_save_name)
            / ".inflight"
            / f"pid-{os.getpid()}-{task_digest}"
        )
        args = argparse.Namespace(
            mode=self.mode,
            black_checkpoint=str(task["black_checkpoint"]),
            white_checkpoint=str(task["white_checkpoint"]),
            num_games=1,
            save_name=inflight_save_name,
            seed=int(task["seed"]),
            game_index_offset=int(task["game_index_offset"]),
            num_workers=int(task.get("num_workers", 1)),
            max_moves=(
                int(task["max_moves"])
                if task.get("max_moves") is not None
                else None
            ),
            num_simulations=(
                int(task["num_simulations"])
                if task.get("num_simulations") is not None
                else None
            ),
            leaf_batch_size=(
                int(task["leaf_batch_size"])
                if task.get("leaf_batch_size") is not None
                else None
            ),
            disable_pcr=bool(task.get("disable_pcr", False)),
        )
        with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                print(
                    f"\n[{utc_timestamp()}] persistent_worker_pid={os.getpid()} "
                    f"physical_gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                    f"task_id={task['task_id']} game_index={args.game_index_offset}",
                    flush=True,
                )
                _run_bounded_games(
                    args,
                    self.config,
                    agent_name_cache=self.agent_name_cache,
                )
        output_dir = GAME_DATA_ROOT / inflight_save_name
        matches = sorted(
            output_dir.glob(f"*-game{args.game_index_offset:07d}.npz")
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one completed game file for index "
                f"{args.game_index_offset}, found {len(matches)}"
            )
        with np.load(matches[0], allow_pickle=False) as data:
            winner = int(data["winner"])
        published_dir = GAME_DATA_ROOT / published_save_name
        published_dir.mkdir(parents=True, exist_ok=True)
        published_path = published_dir / matches[0].name
        if published_path.exists():
            raise FileExistsError(
                f"refusing to replace completed game: {published_path}"
            )
        with open(matches[0], "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(matches[0], published_path)
        parent_fd = os.open(published_dir, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        output_dir.rmdir()
        try:
            output_dir.parent.rmdir()
        except OSError:
            pass
        return {
            "game_path": str(published_path),
            "winner": winner,
        }

    def close(self) -> None:
        from alpha_go import self_play

        self_play._cleanup_thread_local_agents()


def persistent_worker_loop(
    mode: str,
    *,
    input_stream: TextIO | None = None,
) -> int:
    """Serve newline-delimited single-game tasks until an explicit shutdown."""
    stream = sys.stdin if input_stream is None else input_stream
    runner = PersistentGameRunner(load_config(), mode)
    print(
        json.dumps(
            {
                "event": "persistent_worker_ready",
                "mode": mode,
                "pid": os.getpid(),
                "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            task = json.loads(raw_line)
            if task.get("command") == "shutdown":
                break
            task_id = str(task["task_id"])
            response_path = Path(str(task["response_path"]))
            error: str | None = None
            task_result: dict[str, object] = {}
            try:
                task_result = runner.run_task(task) or {}
            except BaseException:
                error = traceback.format_exc()
                log_value = task.get("log_path")
                if log_value:
                    log_path = Path(str(log_value))
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write("\n[FATAL] persistent worker task failed\n")
                        handle.write(error)
                        handle.flush()
            atomic_write_json(
                response_path,
                {
                    "task_id": task_id,
                    "ok": error is None,
                    "error": error,
                    "pid": os.getpid(),
                    **task_result,
                },
            )
    finally:
        runner.close()
    return 0


def main() -> None:
    if "--persistent-worker" in sys.argv[1:]:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--persistent-worker", action="store_true")
        worker_parser.add_argument(
            "--mode", choices=("collect", "arena"), required=True
        )
        worker_args = worker_parser.parse_args()
        raise SystemExit(persistent_worker_loop(worker_args.mode))

    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "arena"), required=True)
    parser.add_argument("--black-checkpoint", required=True)
    parser.add_argument("--white-checkpoint", required=True)
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--save-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--game-index-offset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--leaf-batch-size", type=int, default=None)
    parser.add_argument("--disable-pcr", action="store_true")
    _run_bounded_games(parser.parse_args(), cfg)


if __name__ == "__main__":
    main()
