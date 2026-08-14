#!/usr/bin/env python
"""Persistent, fail-closed controller for the confirmed 9x9 experiment."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from arena import (
    _trajectory_digest,
    evaluate_arena,
    gatekeeper_decision,
)
from bootstrap import extract_game_index
from common import (
    CHECKPOINT_ROOT,
    CONFIG_PATH,
    EXP_DATA_ROOT,
    EXP_DIR,
    EXP_NAME,
    GAME_DATA_ROOT,
    GpuBusyError,
    LOGS_DIR,
    MANIFEST_DIR,
    REPO_ROOT,
    RUNTIME_DIR,
    STATE_PATH,
    STOP_PATH,
    append_event,
    atomic_write_json,
    atomic_write_text,
    disk_free_gib,
    ensure_runtime_dirs,
    load_config,
    preflight_gpus,
    publish_directory,
    relative_data_path,
    sha256_file,
    stop_requested,
    utc_timestamp,
    validate_data_dir,
    validate_npz,
)

PYTHON = Path(os.environ.get("AUTOGO_PYTHON", sys.executable)).resolve()


def _summarize_worker_error(error: object) -> str:
    """Return the actionable exception line instead of a trailing CUDA hint."""
    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    for marker in (
        "CUDA error: out of memory",
        "CUDA out of memory",
        "OutOfMemoryError",
    ):
        for line in reversed(lines):
            if marker in line:
                return line
    return lines[-1] if lines else ""


def _freeze_initial_champion(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and atomically freeze the configured warm-start checkpoint."""
    initial = config["initial_model"]
    source = (REPO_ROOT / str(initial["source_checkpoint"])).resolve()
    if not source.is_relative_to(REPO_ROOT) or not source.is_file():
        raise FileNotFoundError(f"warm-start checkpoint is missing: {source}")
    expected_sha256 = str(initial["source_sha256"])
    source_sha256 = sha256_file(source)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"warm-start checkpoint hash mismatch: {source_sha256} != {expected_sha256}"
        )

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("completed") is not True:
        raise ValueError("warm-start checkpoint is not marked completed")
    if payload.get("model_config") != config["model"]:
        raise ValueError("warm-start checkpoint model_config does not match config.json")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict) or "input_conv.weight" not in state_dict:
        raise ValueError("warm-start checkpoint is missing input_conv.weight")
    input_weight = state_dict["input_conv.weight"]
    if tuple(input_weight.shape[:2]) != (
        int(config["model"]["channels"]),
        4,
    ):
        raise ValueError(
            f"warm-start input tensor has incompatible shape {tuple(input_weight.shape)}"
        )

    frozen = CHECKPOINT_ROOT / str(initial["frozen_checkpoint_name"])
    if frozen.exists():
        if sha256_file(frozen) != expected_sha256:
            raise ValueError(f"existing frozen champion has the wrong hash: {frozen}")
    else:
        temporary = frozen.with_name(frozen.name + ".tmp")
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        if sha256_file(temporary) != expected_sha256:
            temporary.unlink()
            raise ValueError("atomic warm-start copy failed hash verification")
        os.replace(temporary, frozen)
        parent_fd = os.open(frozen.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    return {
        "iteration": 0,
        "checkpoint": str(frozen),
        "sha256": expected_sha256,
        "promoted_at": utc_timestamp(),
        "reason": "released_champion_warm_start",
        "source_checkpoint": str(source),
        "source_experiment": str(initial["source_experiment"]),
        "source_iteration": int(initial["source_iteration"]),
    }


@dataclass(frozen=True)
class ProcessOutcome:
    """Completed task from a dynamic process slot."""

    task: Any
    gpu_id: int
    process: Any
    returncode: int


@dataclass(frozen=True)
class ActiveProcess:
    """Task that is still running when an irreversible result is reached."""

    task: Any
    gpu_id: int
    process: Any


@dataclass
class ArenaGateProgress:
    """Incremental strength and immutable health evidence for early stopping."""

    max_games: int
    required_score: float
    draw_score: float
    minimum_unique_rate: float
    minimum_d4_rate: float
    games: int = 0
    candidate_points: float = 0.0
    trajectory_digests: set[tuple[int, str]] = field(default_factory=set)
    recorded_paths: set[Path] = field(default_factory=set)

    @property
    def decision(self) -> str | None:
        return gatekeeper_decision(
            self.candidate_points,
            self.games,
            self.max_games,
            self.required_score,
        )

    @property
    def can_stop(self) -> bool:
        decision = self.decision
        if decision == "reject":
            return True
        if decision != "accept":
            return False
        required_unique = self.minimum_unique_rate * self.max_games
        return (
            len(self.trajectory_digests) + 1e-12 >= required_unique
            and self.minimum_d4_rate == 0.0
        )

    def record_file(self, job: dict[str, Any], path: Path) -> None:
        resolved = path.resolve()
        if resolved in self.recorded_paths:
            raise ValueError(f"arena gatekeeper game recorded twice: {resolved}")
        side = int(job["side"])
        candidate_color = 1 if side == 0 else 2
        with np.load(resolved, allow_pickle=False) as data:
            winner = int(data["winner"])
            moves = np.asarray(data["moves"])
        if winner not in (0, 1, 2):
            raise ValueError(f"invalid arena winner {winner}: {resolved}")
        if winner == candidate_color:
            self.candidate_points += 1.0
        elif winner == 0:
            self.candidate_points += self.draw_score
        self.games += 1
        if self.games > self.max_games:
            raise ValueError("arena gatekeeper recorded too many games")
        self.trajectory_digests.add((side, _trajectory_digest(moves)))
        self.recorded_paths.add(resolved)

    def snapshot(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "candidate_points": self.candidate_points,
            "decision": self.decision,
            "unique_trajectories": len(self.trajectory_digests),
            "remaining_games_skipped": self.max_games - self.games,
        }


@dataclass
class PersistentTaskProcess:
    """One task assigned to a long-lived worker process."""

    worker: Any
    response_path: Path
    game_log_path: Path
    task_id: str
    result: dict[str, Any] | None = None

    @property
    def pid(self) -> int:
        return int(self.worker.pid)

    def poll(self) -> int | None:
        if self.result is not None:
            return 0 if bool(self.result.get("ok")) else 1
        if self.response_path.exists():
            try:
                payload = json.loads(self.response_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                self.result = {
                    "ok": False,
                    "error": f"invalid worker response {self.response_path}: {exc}",
                }
                return 1
            if payload.get("task_id") != self.task_id:
                self.result = {
                    "ok": False,
                    "error": (
                        f"worker response task {payload.get('task_id')} "
                        f"!= {self.task_id}"
                    ),
                }
                return 1
            self.result = payload
            return 0 if bool(payload.get("ok")) else 1
        worker_returncode = self.worker.poll()
        if worker_returncode is None:
            return None
        self.result = {
            "ok": False,
            "error": (
                f"persistent worker pid={self.worker.pid} exited with "
                f"code={worker_returncode} before responding"
            ),
        }
        return int(worker_returncode) if int(worker_returncode) != 0 else 1


def run_dynamic_process_pool(
    *,
    tasks: list[Any],
    gpu_slots: list[int],
    launch: Callable[[Any, int], Any],
    on_complete: Callable[[ProcessOutcome], None],
    should_stop: Callable[[], bool],
    on_snapshot: Callable[[list[int]], None],
    should_cancel_active: Callable[[], bool] | None = None,
    cancel_active: Callable[[list[ActiveProcess]], None] | None = None,
    poll_interval_seconds: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[ProcessOutcome], int]:
    """Keep every process slot busy until tasks finish or draining is requested."""
    if not gpu_slots:
        raise ValueError("gpu_slots must not be empty")

    pending = deque(tasks)
    available = deque(gpu_slots)
    active: dict[int, tuple[Any, int, Any]] = {}
    outcomes: list[ProcessOutcome] = []
    canceled_games = 0
    canceling = (
        should_cancel_active is not None and should_cancel_active()
    )
    if canceling and cancel_active is None:
        raise ValueError("cancel_active is required when cancellation is enabled")
    draining = should_stop() or canceling

    def fill_available_slots() -> None:
        while pending and available and not draining:
            task = pending.popleft()
            gpu_id = available.popleft()
            process = launch(task, gpu_id)
            if process.pid in active:
                raise RuntimeError(f"duplicate active process pid {process.pid}")
            active[process.pid] = (task, gpu_id, process)

    def cancel_running_tasks() -> None:
        nonlocal canceled_games
        if not active:
            return
        if cancel_active is None:
            raise ValueError("cancel_active callback is missing")
        running = [
            ActiveProcess(task=task, gpu_id=gpu_id, process=process)
            for task, gpu_id, process in active.values()
        ]
        cancel_active(running)
        canceled_games += len(running)
        for item in running:
            available.append(item.gpu_id)
        active.clear()

    fill_available_slots()
    on_snapshot(sorted(active))

    while active:
        completed: list[tuple[int, Any, int, Any, int]] = []
        for pid, (task, gpu_id, process) in list(active.items()):
            returncode = process.poll()
            if returncode is not None:
                completed.append((pid, task, gpu_id, process, int(returncode)))

        if not completed:
            canceling = (
                should_cancel_active is not None and should_cancel_active()
            )
            if canceling:
                cancel_running_tasks()
                on_snapshot([])
                break
            draining = draining or should_stop()
            sleep(poll_interval_seconds)
            continue

        failed = False
        for pid, task, gpu_id, process, returncode in completed:
            active.pop(pid)
            available.append(gpu_id)
            outcome = ProcessOutcome(task, gpu_id, process, returncode)
            on_complete(outcome)
            outcomes.append(outcome)
            failed = failed or returncode != 0

        canceling = (
            should_cancel_active is not None and should_cancel_active()
        )
        draining = draining or failed or should_stop() or canceling
        if canceling:
            cancel_running_tasks()
        else:
            fill_available_slots()
        on_snapshot(sorted(active))

    return outcomes, len(pending) + canceled_games


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _split_count(total: int, chunks: int) -> list[int]:
    base, remainder = divmod(total, chunks)
    return [base + (1 if index < remainder else 0) for index in range(chunks)]


def _chunk_distribution(matchups: int, target_jobs: int) -> list[int]:
    if target_jobs <= matchups:
        return [1] * matchups
    base, remainder = divmod(target_jobs, matchups)
    return [base + (1 if index < remainder else 0) for index in range(matchups)]


def _gpu_process_slots(gpu_ids: list[int], processes_per_gpu: int) -> list[int]:
    """Return an interleaved list with exactly N process slots per GPU."""
    if processes_per_gpu < 1:
        raise ValueError("processes_per_gpu must be positive")
    return [gpu_id for _ in range(processes_per_gpu) for gpu_id in gpu_ids]


def _chunk_missing_game_ranges(
    total_games: int,
    completed_indices: set[int],
    games_per_process: int,
) -> list[tuple[int, int]]:
    """Return contiguous missing game ranges without crossing resume gaps."""
    if games_per_process < 1:
        raise ValueError("games_per_process must be positive")
    chunks: list[tuple[int, int]] = []
    game_index = 0
    while game_index < total_games:
        if game_index in completed_indices:
            game_index += 1
            continue
        start = game_index
        count = 0
        while (
            game_index < total_games
            and game_index not in completed_indices
            and count < games_per_process
        ):
            game_index += 1
            count += 1
        chunks.append((start, count))
    return chunks


def _interleave_arena_game_tasks(
    game_tasks: list[tuple[dict[str, Any], int, int]],
) -> list[tuple[dict[str, Any], int, int]]:
    """Alternate candidate colors while preserving each side's game order."""
    sides: dict[int, deque[tuple[dict[str, Any], int, int]]] = {
        0: deque(),
        1: deque(),
    }
    for task in game_tasks:
        side = int(task[0].get("side", -1))
        if side not in sides:
            raise ValueError("arena game task is missing a valid side")
        sides[side].append(task)

    ordered: list[tuple[dict[str, Any], int, int]] = []
    while sides[0] or sides[1]:
        for side in (0, 1):
            if sides[side]:
                ordered.append(sides[side].popleft())
    return ordered


def _training_gpu_ids(config: dict[str, Any]) -> list[int]:
    training = config["training"]
    values = training.get("gpu_ids")
    gpu_ids = (
        [int(value) for value in values]
        if values is not None
        else [int(training["gpu_id"])]
    )
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("training gpu_ids must be non-empty and unique")
    return gpu_ids


def build_run_games_command(
    job: dict[str, Any],
    *,
    completed_games: int,
    save_name: str,
    games_to_run: int | None = None,
) -> list[str]:
    """Build one resumable collector command from an immutable job spec."""
    remaining = int(job["games"]) - completed_games
    requested = remaining if games_to_run is None else games_to_run
    if requested < 1 or requested > remaining:
        raise ValueError(
            f"games_to_run={requested} outside remaining range 1..{remaining}"
        )
    return [
        str(PYTHON),
        str(EXP_DIR / "run_games.py"),
        "--mode", str(job["mode"]),
        "--black-checkpoint", str(job["black"]),
        "--white-checkpoint", str(job["white"]),
        "--num-games", str(requested),
        "--num-workers", str(int(job["workers"])),
        "--save-name", save_name,
        "--seed", str(int(job["seed"]) + completed_games),
        "--game-index-offset", str(int(job["offset"]) + completed_games),
    ]


def _game_log_path(
    job: dict[str, Any],
    game_index: int,
    game_count: int,
) -> Path:
    base_log = Path(job["log"])
    global_start = int(job["offset"]) + game_index
    global_end = global_start + game_count - 1
    game_slug = (
        f"game{global_start:07d}"
        if game_count == 1
        else f"games{global_start:07d}-{global_end:07d}"
    )
    return base_log.with_name(f"{base_log.stem}-{game_slug}{base_log.suffix}")


class GracefulStop(RuntimeError):
    pass


class Controller:
    def __init__(self, *, once: bool, resume: bool) -> None:
        self.config = load_config()
        self.once = once
        self.resume = resume
        ensure_runtime_dirs()
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        config_digest = sha256_file(CONFIG_PATH)
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text())
            if state["config_sha256"] != config_digest:
                raise RuntimeError(
                    "config.json changed after experiment initialization; refusing to alter provenance"
                )
            if self.resume:
                if STOP_PATH.exists():
                    STOP_PATH.unlink()
                state["stage"] = "PREPARING"
                state["graceful_stop_requested"] = False
                state["last_error"] = None
                self._save_state(state)
            elif state.get("stage") in ("RUNNING", "PREPARING", "COLLECTING", "TRAINING", "ARENA"):
                pass
            elif state.get("stage") in ("STOPPED", "FAILED"):
                raise RuntimeError(
                    f"experiment state is {state['stage']}; use launcher.sh --resume"
                )
            return state

        status = _git(["status", "--porcelain=v1"])
        state = {
            "schema_version": 1,
            "experiment_name": EXP_NAME,
            "config_sha256": config_digest,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_status_at_creation": status.splitlines(),
            "source_template": "experiments/2026-07-25_21-20-local-19x19-full-001",
            "python": str(PYTHON),
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "controller_pid": os.getpid(),
            "tmux": os.environ.get("TMUX", ""),
            "iteration": 0,
            "stage": "PREPARING",
            "champion_iteration": None,
            "champion_checkpoint": None,
            "champion_history": [],
            "candidate_checkpoint": None,
            "manifest": None,
            "manifest_sha256": None,
            "expected_games": 0,
            "completed_games": 0,
            "completed_positions": 0,
            "child_pids": [],
            "last_arena": None,
            "last_promotion": None,
            "last_error": None,
            "recovery_command": None,
            "graceful_stop_requested": False,
            "completed_iterations": [],
        }
        warm_start = None
        if "initial_model" in self.config:
            warm_start = _freeze_initial_champion(self.config)
            atomic_write_json(RUNTIME_DIR / "champion.json", warm_start)
            state.update({
                "iteration": 1,
                "champion_iteration": 0,
                "champion_checkpoint": warm_start["checkpoint"],
                "champion_history": [warm_start],
                "last_promotion": warm_start,
                "initial_model": warm_start,
            })
        self._save_state(state)
        append_event(
            "experiment_initialized",
            config_sha256=config_digest,
            git_commit=state["git_commit"],
            git_dirty=bool(status),
            warm_start=warm_start,
        )
        return state

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at"] = utc_timestamp()
        self.state["controller_pid"] = os.getpid()
        atomic_write_json(STATE_PATH, self.state)

    def _set_stage(self, stage: str, **fields: Any) -> None:
        previous = self.state.get("stage")
        self.state.update(fields)
        self.state["stage"] = stage
        self.state["stage_started_at"] = utc_timestamp()
        self._save_state()
        append_event(
            "stage_changed",
            iteration=self.state["iteration"],
            previous=previous,
            stage=stage,
            **fields,
        )
        print(f"\n=== iteration {self.state['iteration']:04d} stage {stage} ===", flush=True)

    def _fail(self, exc: BaseException) -> None:
        recovery = f"bash {EXP_DIR / 'launcher.sh'} --resume"
        self.state.update({
            "stage": "FAILED",
            "last_error": f"{type(exc).__name__}: {exc}",
            "recovery_command": recovery,
            "child_pids": [],
        })
        self._save_state()
        append_event(
            "experiment_failed",
            iteration=self.state["iteration"],
            error=self.state["last_error"],
            recovery_command=recovery,
        )
        print(f"\nFAILED: {self.state['last_error']}", file=sys.stderr, flush=True)
        print(f"Recovery: {recovery}", file=sys.stderr, flush=True)

    def _stop(self, resume_stage: str) -> None:
        self.state.update({
            "stage": "STOPPED",
            "resume_stage": resume_stage,
            "graceful_stop_requested": True,
            "child_pids": [],
            "recovery_command": f"bash {EXP_DIR / 'launcher.sh'} --resume",
        })
        self._save_state()
        append_event(
            "experiment_stopped",
            iteration=self.state["iteration"],
            resume_stage=resume_stage,
        )
        print("Graceful stop complete; no new work will be scheduled.", flush=True)

    def _check_stop(self, stage: str) -> None:
        if stop_requested():
            raise GracefulStop(stage)

    def _preflight_stage_gpus(
        self,
        gpu_ids: list[int],
        stage: str,
    ) -> dict[str, Any]:
        """Wait for exclusive GPUs without turning temporary contention into failure."""
        require_idle = bool(self.config["health"]["require_idle_gpus"])
        waiting = bool(getattr(self, "state", {}).get("gpu_waiting_since"))
        while True:
            self._check_stop(stage)
            try:
                report = preflight_gpus(gpu_ids, require_idle=require_idle)
            except GpuBusyError as exc:
                if not waiting:
                    self.state.update({
                        "gpu_waiting_since": utc_timestamp(),
                        "gpu_wait_reason": str(exc),
                        "child_pids": [],
                    })
                    self._save_state()
                    append_event(
                        "gpu_preflight_waiting",
                        stage=stage,
                        gpu_ids=gpu_ids,
                        reason=str(exc),
                    )
                    print(
                        f"GPU preflight waiting for {stage}: {exc}",
                        flush=True,
                    )
                    waiting = True
                time.sleep(5.0)
                continue

            if waiting:
                waiting_since = self.state.pop("gpu_waiting_since", None)
                self.state.pop("gpu_wait_reason", None)
                self._save_state()
                append_event(
                    "gpu_preflight_resumed",
                    stage=stage,
                    gpu_ids=gpu_ids,
                    waiting_since=waiting_since,
                )
                print(f"GPU preflight resumed for {stage}", flush=True)
            append_event("gpu_preflight_passed", stage=stage, **report)
            return report

    def _check_disk(self) -> None:
        free = disk_free_gib()
        floor = float(self.config["health"]["disk_free_floor_gib"])
        if free < floor:
            raise RuntimeError(f"disk free {free:.1f} GiB is below floor {floor:.1f} GiB")

    def _run_logged(
        self,
        command: list[str],
        log_path: Path,
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        merged_env = os.environ.copy()
        merged_env.update({
            "GAME_DATA_DIR": str(GAME_DATA_ROOT),
            "AUTOGO_CHECKPOINT_DIR": str(CHECKPOINT_ROOT),
            "AUTOGO_PYTHON": str(PYTHON),
            "PYTHONUNBUFFERED": "1",
        })
        if env:
            merged_env.update(env)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[{utc_timestamp()}] $ {' '.join(command)}\n")
            log.flush()
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=merged_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.state["child_pids"] = [child.pid]
            self._save_state()
            rc = child.wait()
        self.state["child_pids"] = []
        self._save_state()
        return rc

    def _bootstrap(self) -> Path:
        cfg = self.config
        expected = int(cfg["bootstrap"]["games"])
        final = EXP_DATA_ROOT / "bootstrap" / "random-it0000"
        if final.exists():
            validate_data_dir(
                final,
                expected_games=expected,
                board_size=int(cfg["rules"]["board_size"]),
                require_mcts=False,
            )
            return final
        staging = EXP_DATA_ROOT / ".staging" / "bootstrap-random-it0000"
        staging.mkdir(parents=True, exist_ok=True)
        existing = sorted(staging.glob("*.npz"))
        for path in existing:
            validate_npz(
                path,
                board_size=int(cfg["rules"]["board_size"]),
                require_mcts=False,
            )
        done = len(existing)
        if done > expected:
            raise RuntimeError(f"bootstrap staging has {done} files, expected at most {expected}")
        if done < expected:
            self._check_stop("BOOTSTRAP")
            save_name = relative_data_path(staging)
            command = [
                str(PYTHON),
                str(EXP_DIR / "bootstrap.py"),
                "--num-games", str(expected - done),
                "--num-workers", str(cfg["bootstrap"]["workers"]),
                "--save-name", save_name,
                "--seed", str(int(cfg["base_seed"]) + done),
                "--game-index-offset", str(done),
            ]
            rc = self._run_logged(command, LOGS_DIR / "bootstrap.log")
            if rc != 0:
                raise RuntimeError(f"bootstrap collector exited with code {rc}")
        stats = validate_data_dir(
            staging,
            expected_games=expected,
            board_size=int(cfg["rules"]["board_size"]),
            require_mcts=False,
        )
        publish_directory(staging, final)
        append_event("bootstrap_published", path=str(final), **stats)
        return final

    def _collection_matchups(self) -> list[dict[str, Any]]:
        history = self.state["champion_history"][-int(self.config["collection"]["last_k_promoted"]):]
        if not history:
            history = [{
                "iteration": self.state["champion_iteration"],
                "checkpoint": self.state["champion_checkpoint"],
            }]
        current = str(self.state["champion_checkpoint"])
        out: list[dict[str, Any]] = []
        for item in history:
            opponent = str(item["checkpoint"])
            label = f"promoted-it{int(item['iteration']):04d}"
            out.append({
                "name": f"as-black-vs-{label}",
                "black": current,
                "white": opponent,
                "games": int(self.config["collection"]["games_per_matchup"]),
            })
            out.append({
                "name": f"as-white-vs-{label}",
                "black": opponent,
                "white": current,
                "games": int(self.config["collection"]["games_per_matchup"]),
            })
        out.append({
            "name": "selfplay-current-champion",
            "black": current,
            "white": current,
            "games": int(self.config["collection"]["selfplay_games"]),
        })
        return out

    def _collection_jobs(self, iteration: int) -> list[dict[str, Any]]:
        matchups = self._collection_matchups()
        collection = self.config["collection"]
        target_jobs = int(collection["num_jobs"])
        chunks = _chunk_distribution(len(matchups), target_jobs)
        jobs: list[dict[str, Any]] = []
        for matchup_index, (matchup, n_chunks) in enumerate(zip(matchups, chunks)):
            counts = _split_count(int(matchup["games"]), n_chunks)
            offset = 0
            for shard_index, count in enumerate(counts):
                name = f"{matchup['name']}-shard{shard_index:03d}"
                base = Path(f"collect-it{iteration:04d}") / matchup["name"] / f"shard{shard_index:03d}"
                jobs.append({
                    "name": name,
                    "mode": "collect",
                    "black": matchup["black"],
                    "white": matchup["white"],
                    "games": count,
                    "workers": int(self.config["collection"]["workers_per_process"]),
                    "offset": offset,
                    "seed": int(self.config["base_seed"]) + iteration * 10_000_000 + matchup_index * 100_000 + shard_index * 1000,
                    "staging": EXP_DATA_ROOT / ".staging" / base,
                    "final": EXP_DATA_ROOT / base,
                    "log": LOGS_DIR / f"collect-it{iteration:04d}" / f"{name}.log",
                    "require_mcts": True,
                    "allowed_visits": tuple(self.config["mcts"]["pcr_sims"]),
                })
                offset += count
        return jobs

    def _arena_jobs(self, iteration: int, candidate: str, champion: str) -> list[dict[str, Any]]:
        total_jobs = int(self.config["arena"]["num_jobs"])
        if total_jobs % 2:
            raise ValueError("arena num_jobs must be even for color balance")
        per_side_jobs = total_jobs // 2
        specs = [
            (
                "candidate-black",
                candidate,
                champion,
                int(self.config["arena"]["candidate_as_black"]),
                0,
            ),
            (
                "candidate-white",
                champion,
                candidate,
                int(self.config["arena"]["candidate_as_white"]),
                1,
            ),
        ]
        jobs: list[dict[str, Any]] = []
        for label, black, white, games, side in specs:
            counts = _split_count(games, per_side_jobs)
            offset = 0
            for shard_index, count in enumerate(counts):
                name = f"{label}-shard{shard_index:03d}"
                base = Path(f"arena-it{iteration:04d}") / label / f"shard{shard_index:03d}"
                pair_seed = int(self.config["base_seed"]) + 500_000_000 + iteration * 10_000_000 + shard_index * 1000
                seed = (
                    pair_seed
                    if bool(self.config["arena"]["paired_seeds"])
                    else pair_seed + side * 100_000
                )
                jobs.append({
                    "name": name,
                    "mode": "arena",
                    "black": black,
                    "white": white,
                    "games": count,
                    "workers": int(self.config["arena"]["workers_per_process"]),
                    "offset": offset,
                    "seed": seed,
                    "staging": EXP_DATA_ROOT / ".staging" / base,
                    "final": EXP_DATA_ROOT / base,
                    "log": LOGS_DIR / f"arena-it{iteration:04d}" / f"{name}.log",
                    "require_mcts": False,
                    "allowed_visits": (),
                    "side": side,
                })
                offset += count
        return jobs

    def _validate_existing_partial_progress(
        self, job: dict[str, Any]
    ) -> tuple[set[int], int]:
        staging = Path(job["staging"])
        staging.mkdir(parents=True, exist_ok=True)
        files = sorted(staging.glob("*.npz"))
        completed: set[int] = set()
        positions = 0
        for path in files:
            stats = validate_npz(
                path,
                board_size=int(self.config["rules"]["board_size"]),
                require_mcts=bool(job["require_mcts"]),
                allowed_visit_totals=job["allowed_visits"] or (600,),
            )
            positions += int(stats["positions"])
            global_index = extract_game_index(path)
            local_index = global_index - int(job["offset"])
            if local_index < 0 or local_index >= int(job["games"]):
                raise ValueError(
                    f"{job['name']} contains out-of-range game index "
                    f"{global_index}: {path}"
                )
            if local_index in completed:
                raise ValueError(
                    f"{job['name']} contains duplicate game index {global_index}"
                )
            completed.add(local_index)
        if len(files) > int(job["games"]):
            raise RuntimeError(f"{job['name']} has too many partial files: {len(files)}")
        return completed, positions

    def _run_legacy_game_tasks(
        self,
        game_tasks: list[tuple[dict[str, Any], int, int]],
        gpu_slots: list[int],
        omp_threads: int,
    ) -> tuple[list[ProcessOutcome], int]:
        resources: dict[int, Any] = {}

        def launch_game(task: tuple[dict[str, Any], int, int], gpu_id: int) -> Any:
            job, game_index, game_count = task
            save_name = relative_data_path(Path(job["staging"]))
            command = build_run_games_command(
                job,
                completed_games=game_index,
                save_name=save_name,
                games_to_run=game_count,
            )
            env = os.environ.copy()
            env.update({
                "GAME_DATA_DIR": str(GAME_DATA_ROOT),
                "AUTOGO_CHECKPOINT_DIR": str(CHECKPOINT_ROOT),
                "AUTOGO_PYTHON": str(PYTHON),
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(omp_threads),
            })
            log_path = _game_log_path(job, game_index, game_count)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "a", encoding="utf-8")
            handle.write(
                f"\n[{utc_timestamp()}] physical_gpu={gpu_id} "
                f"game_index={game_index} games={game_count} $ {' '.join(command)}\n"
            )
            handle.flush()
            try:
                child = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except BaseException:
                handle.close()
                raise
            child.game_log_path = log_path
            resources[child.pid] = handle
            print(
                f"launched {job['name']} games={game_index + 1}-"
                f"{game_index + game_count}/{job['games']} "
                f"pid={child.pid} physical_gpu={gpu_id}",
                flush=True,
            )
            return child

        def complete_game(outcome: ProcessOutcome) -> None:
            resources.pop(outcome.process.pid).close()

        def save_active_pids(pids: list[int]) -> None:
            self.state["child_pids"] = pids
            self._save_state()

        return run_dynamic_process_pool(
            tasks=game_tasks,
            gpu_slots=gpu_slots,
            launch=launch_game,
            on_complete=complete_game,
            should_stop=stop_requested,
            on_snapshot=save_active_pids,
        )

    def _run_persistent_game_tasks(
        self,
        game_tasks: list[tuple[dict[str, Any], int, int]],
        gpu_slots: list[int],
        omp_threads: int,
        arena_gate: ArenaGateProgress | None = None,
    ) -> tuple[list[ProcessOutcome], int]:
        if any(game_count != 1 for _job, _index, game_count in game_tasks):
            raise ValueError("persistent workers require one game per task")
        if not game_tasks:
            return [], 0
        if arena_gate is not None and arena_gate.can_stop:
            return [], len(game_tasks)

        mode = str(game_tasks[0][0]["mode"])
        iteration = int(self.state["iteration"])
        response_dir = (
            RUNTIME_DIR
            / "worker-responses"
            / f"{mode}-it{iteration:04d}-{os.getpid()}-{time.time_ns()}"
        )
        response_dir.mkdir(parents=True, exist_ok=False)
        worker_log_dir = (
            LOGS_DIR / f"{mode}-it{iteration:04d}" / "persistent-workers"
        )
        worker_log_dir.mkdir(parents=True, exist_ok=True)

        worker_records: list[tuple[Any, Any, int, int, Path]] = []
        canceled_worker_pids: set[int] = set()
        idle_by_gpu: dict[int, deque[Any]] = {
            gpu_id: deque() for gpu_id in sorted(set(gpu_slots))
        }
        slots_seen: dict[int, int] = {}

        try:
            for gpu_id in gpu_slots:
                slot_index = slots_seen.get(gpu_id, 0)
                slots_seen[gpu_id] = slot_index + 1
                worker_log = worker_log_dir / (
                    f"gpu{gpu_id}-slot{slot_index:02d}.log"
                )
                handle = open(worker_log, "a", encoding="utf-8", buffering=1)
                env = os.environ.copy()
                env.update({
                    "GAME_DATA_DIR": str(GAME_DATA_ROOT),
                    "AUTOGO_CHECKPOINT_DIR": str(CHECKPOINT_ROOT),
                    "AUTOGO_PYTHON": str(PYTHON),
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": str(omp_threads),
                })
                try:
                    worker = subprocess.Popen(
                        [
                            str(PYTHON),
                            str(EXP_DIR / "run_games.py"),
                            "--persistent-worker",
                            "--mode",
                            mode,
                        ],
                        cwd=REPO_ROOT,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                except BaseException:
                    handle.close()
                    raise
                if worker.stdin is None:
                    handle.close()
                    raise RuntimeError("persistent worker stdin pipe was not created")
                worker_records.append(
                    (worker, handle, gpu_id, slot_index, worker_log)
                )
                idle_by_gpu[gpu_id].append(worker)

            task_counter = 0

            def launch_game(
                task: tuple[dict[str, Any], int, int], gpu_id: int
            ) -> PersistentTaskProcess:
                nonlocal task_counter
                if not idle_by_gpu[gpu_id]:
                    raise RuntimeError(f"no idle persistent worker for GPU {gpu_id}")
                worker = idle_by_gpu[gpu_id].popleft()
                if worker.poll() is not None:
                    raise RuntimeError(
                        f"persistent worker pid={worker.pid} exited before dispatch"
                    )
                job, game_index, game_count = task
                task_counter += 1
                task_id = (
                    f"{mode}-it{iteration:04d}-{task_counter:05d}-"
                    f"{job['name']}-game{int(job['offset']) + game_index:07d}"
                )
                response_path = response_dir / f"{task_counter:05d}.json"
                game_log = _game_log_path(job, game_index, game_count)
                payload = {
                    "command": "run",
                    "task_id": task_id,
                    "mode": mode,
                    "black_checkpoint": str(job["black"]),
                    "white_checkpoint": str(job["white"]),
                    "save_name": relative_data_path(Path(job["staging"])),
                    "seed": int(job["seed"]) + game_index,
                    "game_index_offset": int(job["offset"]) + game_index,
                    "num_workers": int(job["workers"]),
                    "log_path": str(game_log),
                    "response_path": str(response_path),
                }
                worker.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
                worker.stdin.flush()
                print(
                    f"assigned {job['name']} game={game_index + 1}/"
                    f"{job['games']} worker_pid={worker.pid} "
                    f"physical_gpu={gpu_id}",
                    flush=True,
                )
                return PersistentTaskProcess(
                    worker=worker,
                    response_path=response_path,
                    game_log_path=game_log,
                    task_id=task_id,
                )

            def complete_game(outcome: ProcessOutcome) -> None:
                task_process = outcome.process
                if outcome.returncode == 0 and arena_gate is not None:
                    job, game_index, _game_count = outcome.task
                    result = task_process.result or {}
                    game_path_value = result.get("game_path")
                    if not game_path_value:
                        raise ValueError(
                            f"arena worker {task_process.task_id} omitted game_path"
                        )
                    game_path = Path(str(game_path_value))
                    if game_path.resolve().parent != Path(job["staging"]).resolve():
                        raise ValueError(
                            f"arena worker returned an unexpected path: {game_path}"
                        )
                    expected_index = int(job["offset"]) + game_index
                    if extract_game_index(game_path) != expected_index:
                        raise ValueError(
                            f"arena worker returned game index "
                            f"{extract_game_index(game_path)}, expected {expected_index}"
                        )
                    arena_gate.record_file(job, game_path)
                if task_process.worker.poll() is None:
                    idle_by_gpu[outcome.gpu_id].append(task_process.worker)

            def save_worker_pids(_active_task_pids: list[int]) -> None:
                self.state["child_pids"] = sorted(
                    int(worker.pid)
                    for worker, _handle, _gpu, _slot, _log in worker_records
                    if worker.poll() is None
                )
                self._save_state()

            def cancel_gate_games(active: list[ActiveProcess]) -> None:
                for item in active:
                    worker = item.process.worker
                    canceled_worker_pids.add(int(worker.pid))
                    if worker.poll() is None:
                        try:
                            worker.terminate()
                        except ProcessLookupError:
                            pass
                print(
                    f"gatekeeper locked; canceled {len(active)} "
                    "in-flight games",
                    flush=True,
                )

            save_worker_pids([])
            outcomes, unstarted_games = run_dynamic_process_pool(
                tasks=game_tasks,
                gpu_slots=gpu_slots,
                launch=launch_game,
                on_complete=complete_game,
                should_stop=stop_requested,
                should_cancel_active=(
                    (lambda: arena_gate.can_stop)
                    if arena_gate is not None
                    else None
                ),
                cancel_active=(
                    cancel_gate_games if arena_gate is not None else None
                ),
                on_snapshot=save_worker_pids,
            )
        finally:
            for worker, _handle, _gpu, _slot, _log in worker_records:
                if worker.poll() is None and worker.stdin is not None:
                    try:
                        worker.stdin.write(
                            json.dumps({"command": "shutdown"}) + "\n"
                        )
                        worker.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
            for worker, _handle, _gpu, _slot, _log in worker_records:
                if worker.stdin is not None:
                    try:
                        worker.stdin.close()
                    except OSError:
                        pass
            for worker, handle, _gpu, _slot, worker_log in worker_records:
                if worker.poll() is None:
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        worker.terminate()
                        worker.wait(timeout=10)
                handle.close()
                if (
                    worker.returncode not in (0, None)
                    and int(worker.pid) not in canceled_worker_pids
                ):
                    print(
                        f"persistent worker pid={worker.pid} exited with "
                        f"code={worker.returncode}; log={worker_log}",
                        flush=True,
                    )
            staging_roots = {
                Path(job["staging"])
                for job, _game_index, _game_count in game_tasks
            }
            for staging in staging_roots:
                inflight = staging / ".inflight"
                if inflight.exists():
                    shutil.rmtree(inflight)
            self.state["child_pids"] = []
            self._save_state()

        return outcomes, unstarted_games

    def _run_gpu_jobs_without_early_gate(
        self,
        jobs: list[dict[str, Any]],
        gpu_ids: list[int],
        stage: str,
    ) -> list[Path]:
        cfg = self.config
        pending: list[dict[str, Any]] = []
        positions = 0
        completed_games = 0
        for job in jobs:
            final = Path(job["final"])
            if final.exists():
                stats = validate_data_dir(
                    final,
                    expected_games=int(job["games"]),
                    board_size=int(cfg["rules"]["board_size"]),
                    require_mcts=bool(job["require_mcts"]),
                    allowed_visit_totals=job["allowed_visits"] or (600,),
                )
                completed_games += stats["games"]
                positions += stats["positions"]
            else:
                pending.append(job)
        self.state.update({
            "expected_games": sum(int(job["games"]) for job in jobs),
            "completed_games": completed_games,
            "completed_positions": positions,
        })
        self._save_state()
        stage_config = cfg["arena"] if jobs and jobs[0]["mode"] == "arena" else cfg["collection"]
        games_per_process = int(stage_config.get("games_per_process", 1))
        game_tasks: list[tuple[dict[str, Any], int, int]] = []
        staged_games = 0
        staged_positions = 0
        for job in pending:
            completed_indices, job_positions = (
                self._validate_existing_partial_progress(job)
            )
            staged_games += len(completed_indices)
            staged_positions += job_positions
            game_tasks.extend(
                (job, game_index, game_count)
                for game_index, game_count in _chunk_missing_game_ranges(
                    int(job["games"]), completed_indices, games_per_process
                )
            )
        self.state.update(
            completed_games=completed_games + staged_games,
            completed_positions=positions + staged_positions,
        )
        self._save_state()

        if game_tasks:
            self._preflight_stage_gpus(gpu_ids, stage)
        processes_per_gpu = int(stage_config["processes_per_gpu"])
        omp_threads = int(stage_config["omp_threads_per_process"])
        gpu_slots = _gpu_process_slots(gpu_ids, processes_per_gpu)
        append_event(
            "gpu_process_pool_configured",
            stage=stage,
            scheduling=(
                "persistent_dynamic_refill"
                if bool(stage_config.get("persistent_workers", False))
                else "dynamic_refill"
            ),
            physical_gpus=gpu_ids,
            processes_per_gpu=processes_per_gpu,
            games_per_process=games_per_process,
            persistent_workers=bool(
                stage_config.get("persistent_workers", False)
            ),
            total_process_slots=len(gpu_slots),
            omp_threads_per_process=omp_threads,
            pending_processes=len(game_tasks),
            pending_games=sum(game_count for _job, _start, game_count in game_tasks),
        )
        if bool(stage_config.get("persistent_workers", False)):
            outcomes, unstarted_games = self._run_persistent_game_tasks(
                game_tasks,
                gpu_slots,
                omp_threads,
            )
        else:
            outcomes, unstarted_games = self._run_legacy_game_tasks(
                game_tasks,
                gpu_slots,
                omp_threads,
            )
        failures: list[str] = []
        for outcome in outcomes:
            if outcome.returncode == 0:
                continue
            job, game_index, game_count = outcome.task
            log_path = getattr(outcome.process, "game_log_path", "unknown")
            result = getattr(outcome.process, "result", None) or {}
            error = _summarize_worker_error(result.get("error"))
            error_suffix = f" error={error}" if error else ""
            failures.append(
                f"{job['name']} games={game_index}-{game_index + game_count - 1} "
                f"GPU={outcome.gpu_id} "
                f"code={outcome.returncode} log={log_path}{error_suffix}"
            )

        if failures or unstarted_games:
            staged_games = 0
            staged_positions = 0
            for job in pending:
                completed_indices, job_positions = (
                    self._validate_existing_partial_progress(job)
                )
                staged_games += len(completed_indices)
                staged_positions += job_positions
            self.state.update(
                child_pids=[],
                completed_games=completed_games + staged_games,
                completed_positions=positions + staged_positions,
            )
            self._save_state()
        if failures:
            raise RuntimeError("; ".join(failures))
        if unstarted_games:
            if stop_requested():
                raise GracefulStop(stage)
            raise RuntimeError(
                f"dynamic GPU pool exited with {unstarted_games} unstarted games"
            )

        for job in pending:
            stats = validate_data_dir(
                Path(job["staging"]),
                expected_games=int(job["games"]),
                board_size=int(cfg["rules"]["board_size"]),
                require_mcts=bool(job["require_mcts"]),
                allowed_visit_totals=job["allowed_visits"] or (600,),
            )
            publish_directory(Path(job["staging"]), Path(job["final"]))
            completed_games += stats["games"]
            positions += stats["positions"]
            append_event(
                "shard_published",
                stage=stage,
                iteration=self.state["iteration"],
                name=job["name"],
                physical_gpus=gpu_ids,
                path=str(job["final"]),
                **stats,
            )
        self.state.update(
            child_pids=[],
            completed_games=completed_games,
            completed_positions=positions,
        )
        self._save_state()
        return [Path(job["final"]) for job in jobs]

    def _run_gpu_jobs(
        self,
        jobs: list[dict[str, Any]],
        gpu_ids: list[int],
        stage: str,
    ) -> list[Path]:
        is_early_arena = bool(
            jobs
            and jobs[0]["mode"] == "arena"
            and self.config["arena"].get(
                "early_terminate_irreversible", False
            )
        )
        if not is_early_arena:
            return self._run_gpu_jobs_without_early_gate(jobs, gpu_ids, stage)
        return self._run_arena_gatekeeper_jobs(jobs, gpu_ids, stage)

    def _run_arena_gatekeeper_jobs(
        self,
        jobs: list[dict[str, Any]],
        gpu_ids: list[int],
        stage: str,
    ) -> list[Path]:
        """Run a resumable Arena and publish a mathematically final prefix."""
        cfg = self.config
        arena_cfg = cfg["arena"]
        iteration = int(self.state["iteration"])
        marker_path = RUNTIME_DIR / f"arena-it{iteration:04d}-early-stop.json"
        marker: dict[str, Any] | None = None
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            marker_jobs = marker.get("games_by_job")
            if (
                int(marker.get("max_games", -1)) != int(arena_cfg["games"])
                or float(marker.get("threshold", -1.0))
                != float(arena_cfg["promotion_threshold"])
                or not isinstance(marker_jobs, dict)
                or set(marker_jobs) != {str(job["name"]) for job in jobs}
            ):
                raise ValueError(
                    f"arena early-stop marker does not match config/jobs: "
                    f"{marker_path}"
                )

        def target_games(job: dict[str, Any]) -> int:
            maximum = int(job["games"])
            count = (
                maximum
                if marker is None
                else int(marker["games_by_job"][str(job["name"])])
            )
            if count < 0 or count > maximum:
                raise ValueError(
                    f"arena target for {job['name']} is outside 0..{maximum}"
                )
            return count

        pending: list[dict[str, Any]] = []
        published_games = 0
        published_positions = 0
        staged_games = 0
        staged_positions = 0
        game_tasks: list[tuple[dict[str, Any], int, int]] = []
        for job in jobs:
            final = Path(job["final"])
            staging = Path(job["staging"])
            if final.exists() and staging.exists() and any(staging.glob("*.npz")):
                raise ValueError(
                    f"arena job has both published and staged games: {job['name']}"
                )
            if final.exists():
                stats = validate_data_dir(
                    final,
                    expected_games=target_games(job),
                    board_size=int(cfg["rules"]["board_size"]),
                    require_mcts=False,
                )
                published_games += stats["games"]
                published_positions += stats["positions"]
                continue

            completed_indices, positions = self._validate_existing_partial_progress(
                job
            )
            if marker is not None and len(completed_indices) != target_games(job):
                raise ValueError(
                    f"arena staged games for {job['name']}="
                    f"{len(completed_indices)}, marker={target_games(job)}"
                )
            if target_games(job) == 0:
                if completed_indices:
                    raise ValueError(
                        f"arena marker skips non-empty job {job['name']}"
                    )
                continue
            pending.append(job)
            staged_games += len(completed_indices)
            staged_positions += positions
            if marker is None:
                game_tasks.extend(
                    (job, game_index, game_count)
                    for game_index, game_count in _chunk_missing_game_ranges(
                        int(job["games"]), completed_indices, 1
                    )
                )

        def load_gate_progress() -> ArenaGateProgress:
            progress = ArenaGateProgress(
                max_games=int(arena_cfg["games"]),
                required_score=float(arena_cfg["promotion_threshold"]),
                draw_score=float(arena_cfg["draw_score"]),
                minimum_unique_rate=float(
                    cfg["health"]["arena_min_unique_trajectory_rate"]
                ),
                minimum_d4_rate=float(
                    cfg["health"]["arena_min_d4_opening_unique_rate"]
                ),
            )
            for gate_job in jobs:
                final = Path(gate_job["final"])
                root = (
                    final
                    if final.exists()
                    else Path(gate_job["staging"])
                )
                for game_path in sorted(root.glob("*.npz")):
                    progress.record_file(gate_job, game_path)
            return progress

        gate = load_gate_progress()
        if marker is not None:
            snapshot = gate.snapshot()
            for key in (
                "games",
                "candidate_points",
                "decision",
                "unique_trajectories",
            ):
                if snapshot[key] != marker.get(key):
                    raise ValueError(
                        f"arena marker {key} does not match published/staged data"
                    )

        game_tasks = _interleave_arena_game_tasks(game_tasks)
        expected_games = (
            int(arena_cfg["games"])
            if marker is None
            else sum(int(value) for value in marker["games_by_job"].values())
        )
        self.state.update(
            expected_games=expected_games,
            completed_games=published_games + staged_games,
            completed_positions=published_positions + staged_positions,
        )
        self._save_state()

        if game_tasks and not gate.can_stop:
            self._preflight_stage_gpus(gpu_ids, stage)
        processes_per_gpu = int(arena_cfg["processes_per_gpu"])
        omp_threads = int(arena_cfg["omp_threads_per_process"])
        gpu_slots = _gpu_process_slots(gpu_ids, processes_per_gpu)
        append_event(
            "gpu_process_pool_configured",
            stage=stage,
            scheduling="persistent_dynamic_refill_gatekeeper",
            physical_gpus=gpu_ids,
            processes_per_gpu=processes_per_gpu,
            games_per_process=1,
            persistent_workers=True,
            gatekeeper_early_termination=True,
            total_process_slots=len(gpu_slots),
            omp_threads_per_process=omp_threads,
            pending_processes=len(game_tasks),
            pending_games=len(game_tasks),
        )
        outcomes, unstarted_games = self._run_persistent_game_tasks(
            game_tasks,
            gpu_slots,
            omp_threads,
            arena_gate=gate,
        )
        if unstarted_games and gate.can_stop:
            gate = load_gate_progress()

        failures: list[str] = []
        for outcome in outcomes:
            if outcome.returncode == 0:
                continue
            job, game_index, game_count = outcome.task
            result = getattr(outcome.process, "result", None) or {}
            error = _summarize_worker_error(result.get("error"))
            error_suffix = f" error={error}" if error else ""
            failures.append(
                f"{job['name']} games={game_index}-"
                f"{game_index + game_count - 1} GPU={outcome.gpu_id} "
                f"code={outcome.returncode} "
                f"log={outcome.process.game_log_path}{error_suffix}"
            )

        staged_games = 0
        staged_positions = 0
        for job in pending:
            completed_indices, positions = self._validate_existing_partial_progress(
                job
            )
            staged_games += len(completed_indices)
            staged_positions += positions
        self.state.update(
            child_pids=[],
            completed_games=published_games + staged_games,
            completed_positions=published_positions + staged_positions,
        )
        self._save_state()
        if failures:
            raise RuntimeError("; ".join(failures))
        if unstarted_games and stop_requested():
            raise GracefulStop(stage)
        if unstarted_games and not gate.can_stop:
            raise RuntimeError(
                f"dynamic GPU pool exited with {unstarted_games} unstarted games"
            )

        if unstarted_games and marker is None:
            games_by_job: dict[str, int] = {}
            for job in jobs:
                final = Path(job["final"])
                root = final if final.exists() else Path(job["staging"])
                files = sorted(root.glob("*.npz"))
                for path in files:
                    validate_npz(
                        path,
                        board_size=int(cfg["rules"]["board_size"]),
                        require_mcts=False,
                    )
                games_by_job[str(job["name"])] = len(files)
            marker = {
                "schema_version": 1,
                "iteration": iteration,
                "max_games": int(arena_cfg["games"]),
                "threshold": float(arena_cfg["promotion_threshold"]),
                "candidate_wins_ties": True,
                "games_by_job": games_by_job,
                **gate.snapshot(),
            }
            if sum(games_by_job.values()) != int(marker["games"]):
                raise ValueError(
                    "arena gatekeeper progress disagrees with completed files"
                )
            atomic_write_json(marker_path, marker)
            append_event(
                "arena_gate_decided_early",
                iteration=iteration,
                marker=str(marker_path),
                canceled_or_unstarted_games=unstarted_games,
                **gate.snapshot(),
            )

        if marker is not None:
            expected_games = sum(
                int(value) for value in marker["games_by_job"].values()
            )
        for job in pending:
            publish_games = target_games(job)
            if publish_games == 0:
                continue
            stats = validate_data_dir(
                Path(job["staging"]),
                expected_games=publish_games,
                board_size=int(cfg["rules"]["board_size"]),
                require_mcts=False,
            )
            publish_directory(Path(job["staging"]), Path(job["final"]))
            append_event(
                "shard_published",
                stage=stage,
                iteration=iteration,
                name=job["name"],
                physical_gpus=gpu_ids,
                partial=publish_games < int(job["games"]),
                path=str(job["final"]),
                **stats,
            )

        completed_positions = 0
        completed_games = 0
        published_paths: list[Path] = []
        for job in jobs:
            final = Path(job["final"])
            if not final.exists():
                continue
            stats = validate_data_dir(
                final,
                expected_games=target_games(job),
                board_size=int(cfg["rules"]["board_size"]),
                require_mcts=False,
            )
            completed_games += stats["games"]
            completed_positions += stats["positions"]
            published_paths.append(final)
        self.state.update(
            child_pids=[],
            expected_games=expected_games,
            completed_games=completed_games,
            completed_positions=completed_positions,
        )
        self._save_state()
        return published_paths

    def _build_manifest(self, iteration: int, new_dirs: list[Path]) -> Path:
        lines: list[str] = []
        if iteration > 1:
            previous = MANIFEST_DIR / f"dataset-it{iteration-1:04d}.txt"
            if not previous.exists():
                raise FileNotFoundError(f"previous cumulative manifest missing: {previous}")
            lines.extend(
                line.strip()
                for line in previous.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
        lines.extend(relative_data_path(path) for path in new_dirs)
        unique = list(dict.fromkeys(lines))
        body = (
            f"# {EXP_NAME} immutable dataset for candidate iter{iteration:04d}\n"
            f"# generated {utc_timestamp()}\n"
            + "\n".join(unique)
            + "\n"
        )
        manifest = MANIFEST_DIR / f"dataset-it{iteration:04d}.txt"
        if manifest.exists() and manifest.read_text() != body:
            old_data = [
                line for line in manifest.read_text().splitlines()
                if line and not line.startswith("#")
            ]
            if old_data != unique:
                raise RuntimeError(f"immutable manifest content changed: {manifest}")
            body = manifest.read_text()
        else:
            atomic_write_text(manifest, body)
        digest = sha256_file(manifest)
        self.state.update({"manifest": str(manifest), "manifest_sha256": digest})
        self._save_state()
        append_event(
            "manifest_published",
            iteration=iteration,
            path=str(manifest),
            sha256=digest,
            directories=len(unique),
        )
        return manifest

    def _validate_checkpoint(self, path: Path, iteration: int, manifest: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
            "iteration",
            "step",
            "manifest_sha256",
            "completed",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"checkpoint {path} missing keys {missing}")
        if int(payload["iteration"]) != iteration:
            raise ValueError(f"checkpoint iteration {payload['iteration']} != {iteration}")
        if payload["manifest_sha256"] != sha256_file(manifest):
            raise ValueError("checkpoint manifest digest mismatch")
        if not bool(payload["completed"]):
            raise ValueError("candidate checkpoint is not marked completed")
        for name, tensor in payload["model_state_dict"].items():
            if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"checkpoint tensor {name} contains NaN/Inf")
        return payload

    def _train_candidate(self, iteration: int, manifest: Path, source: str) -> Path:
        candidate = CHECKPOINT_ROOT / f"iter{iteration:04d}-candidate.pt"
        if candidate.exists():
            self._validate_checkpoint(candidate, iteration, manifest)
            return candidate
        training_args = [
            str(PYTHON),
            str(EXP_DIR / "train.py"),
            "--manifest", str(manifest),
            "--iteration", str(iteration),
            "--max-steps",
            str(int(self.config["training"]["max_steps_per_iteration"])),
        ]
        if source:
            training_args.extend(["--source-champion", source])
        gpu_ids = _training_gpu_ids(self.config)
        if len(gpu_ids) > 1:
            command = [
                str(PYTHON),
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={len(gpu_ids)}",
                *training_args[1:],
            ]
        else:
            command = training_args
        self._preflight_stage_gpus(gpu_ids, "TRAINING")
        rc = self._run_logged(
            command,
            LOGS_DIR / f"train-it{iteration:04d}.log",
            env={
                "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "OMP_NUM_THREADS": str(
                    int(self.config["training"].get("omp_threads_per_rank", 4))
                ),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            },
        )
        if rc != 0:
            raise RuntimeError(
                f"training failed with code {rc}; log={LOGS_DIR / f'train-it{iteration:04d}.log'}"
            )
        payload = self._validate_checkpoint(candidate, iteration, manifest)
        append_event(
            "candidate_checkpoint_validated",
            iteration=iteration,
            path=str(candidate),
            step=int(payload["step"]),
        )
        return candidate

    def _promote_initial(self, candidate: Path) -> None:
        pointer = {
            "iteration": 0,
            "checkpoint": str(candidate),
            "promoted_at": utc_timestamp(),
            "reason": "initial_bootstrap_champion",
        }
        atomic_write_json(RUNTIME_DIR / "champion.json", pointer)
        self.state.update({
            "champion_iteration": 0,
            "champion_checkpoint": str(candidate),
            "champion_history": [pointer],
            "candidate_checkpoint": str(candidate),
            "last_promotion": pointer,
        })
        self._save_state()
        append_event("initial_champion_created", **pointer)
        print("=" * 72)
        print("PROMOTED: iter0000 bootstrap candidate -> initial champion")
        print(f"Checkpoint: {candidate}")
        print("=" * 72, flush=True)

    def _decide_arena(self, iteration: int, candidate: Path, result: dict[str, Any]) -> None:
        champion_before = str(self.state["champion_checkpoint"])
        if result["promoted"]:
            pointer = {
                "iteration": iteration,
                "checkpoint": str(candidate),
                "promoted_at": utc_timestamp(),
                "arena_score": result["score"],
                "reason": result["decision_reason"],
            }
            atomic_write_json(RUNTIME_DIR / "champion.json", pointer)
            history = list(self.state["champion_history"])
            history.append(pointer)
            self.state.update({
                "champion_iteration": iteration,
                "champion_checkpoint": str(candidate),
                "champion_history": history,
                "last_promotion": pointer,
            })
            self._save_state()
            append_event(
                "candidate_promoted",
                iteration=iteration,
                old_champion=champion_before,
                new_champion=str(candidate),
                **result,
            )
            print("=" * 72)
            print(f"PROMOTED: iter{iteration:04d} candidate -> new champion")
            print(
                f"Arena: {result['wins']}-{result['losses']}-{result['draws']}, "
                f"score={result['score']:.1%}, threshold={result['threshold']:.1%}"
            )
            print(f"Checkpoint: {candidate}")
            print("=" * 72, flush=True)
        else:
            append_event(
                "candidate_rejected",
                iteration=iteration,
                retained_champion=champion_before,
                candidate=str(candidate),
                **result,
            )
            print("=" * 72)
            print(f"REJECTED: iter{iteration:04d} candidate; champion retained")
            print(
                f"Arena: {result['wins']}-{result['losses']}-{result['draws']}, "
                f"score={result['score']:.1%}, threshold={result['threshold']:.1%}"
            )
            print(f"Reason: {result['decision_reason']}")
            print("=" * 72, flush=True)

    def run(self) -> None:
        append_event(
            "controller_started",
            pid=os.getpid(),
            once=self.once,
            resume=self.resume,
            python=str(PYTHON),
        )
        while True:
            self._check_disk()
            self._check_stop(self.state.get("stage", "PREPARING"))
            iteration = int(self.state["iteration"])
            if self.state["champion_checkpoint"] is None:
                self._set_stage("BOOTSTRAP")
                bootstrap_dir = self._bootstrap()
                self._set_stage("BUILDING_MANIFEST")
                manifest = self._build_manifest(0, [bootstrap_dir])
                self._set_stage("TRAINING", manifest=str(manifest))
                candidate = self._train_candidate(0, manifest, "")
                self._check_stop("VALIDATING_CHECKPOINT")
                self._set_stage("VALIDATING_CHECKPOINT", candidate_checkpoint=str(candidate))
                self._validate_checkpoint(candidate, 0, manifest)
                self._promote_initial(candidate)
                self.state["completed_iterations"] = [0]
                self.state["iteration"] = 1
                self.state["stage"] = "PREPARING"
                self._save_state()
                if self.once:
                    self._set_stage("IDLE")
                    return
                continue

            self._set_stage("COLLECTING")
            collect_jobs = self._collection_jobs(iteration)
            new_dirs = self._run_gpu_jobs(
                collect_jobs,
                [int(x) for x in self.config["collection"]["gpu_ids"]],
                "COLLECTING",
            )
            self._set_stage("BUILDING_MANIFEST")
            manifest = self._build_manifest(iteration, new_dirs)
            self._set_stage("TRAINING", manifest=str(manifest))
            source = str(self.state["champion_checkpoint"])
            candidate = self._train_candidate(iteration, manifest, source)
            self._check_stop("ARENA")
            self._set_stage("ARENA")
            arena_jobs = self._arena_jobs(iteration, str(candidate), source)
            self._run_gpu_jobs(
                arena_jobs,
                [int(x) for x in self.config["arena"]["gpu_ids"]],
                "ARENA",
            )
            black_root = EXP_DATA_ROOT / f"arena-it{iteration:04d}" / "candidate-black"
            white_root = EXP_DATA_ROOT / f"arena-it{iteration:04d}" / "candidate-white"
            result = evaluate_arena(black_root, white_root)
            arena_result_path = RUNTIME_DIR / f"arena-it{iteration:04d}.json"
            atomic_write_json(arena_result_path, result)
            self.state["last_arena"] = {**result, "path": str(arena_result_path)}
            self._save_state()
            self._set_stage("PROMOTED" if result["promoted"] else "REJECTED")
            self._decide_arena(iteration, candidate, result)
            completed = list(self.state["completed_iterations"])
            if iteration not in completed:
                completed.append(iteration)
            self.state.update({
                "completed_iterations": completed,
                "iteration": iteration + 1,
                "stage": "PREPARING",
                "candidate_checkpoint": str(candidate),
                "child_pids": [],
            })
            self._save_state()
            if self.once:
                self._set_stage("IDLE")
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    ensure_runtime_dirs()
    lock_path = RUNTIME_DIR / "controller.lock"
    lock_handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another controller owns {lock_path}") from exc

    controller = Controller(once=args.once, resume=args.resume)
    if args.preflight_only:
        gpu_ids = sorted({
            *[int(x) for x in controller.config["collection"]["gpu_ids"]],
            *_training_gpu_ids(controller.config),
            *[int(x) for x in controller.config["arena"]["gpu_ids"]],
        })
        report = preflight_gpus(
            gpu_ids,
            require_idle=bool(controller.config["health"]["require_idle_gpus"]),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    try:
        controller.run()
    except GracefulStop as exc:
        controller._stop(str(exc))
    except BaseException as exc:
        traceback.print_exc()
        controller._fail(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
