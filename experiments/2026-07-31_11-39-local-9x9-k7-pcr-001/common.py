"""Shared paths, atomic I/O, validation, and GPU preflight helpers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP_NAME = EXP_DIR.name
CONFIG_PATH = EXP_DIR / "config.json"
RUNTIME_DIR = EXP_DIR / "runtime"
LOGS_DIR = EXP_DIR / "logs"
MANIFEST_DIR = EXP_DIR / "manifests"
GAME_DATA_ROOT = Path(
    os.environ.get("GAME_DATA_DIR", REPO_ROOT / "local_data" / "game_data")
).resolve()
EXP_DATA_ROOT = GAME_DATA_ROOT / "experiments" / EXP_NAME
CHECKPOINT_ROOT = Path(
    os.environ.get(
        "AUTOGO_CHECKPOINT_DIR",
        REPO_ROOT / "local_data" / "checkpoints" / EXP_NAME,
    )
).resolve()
STATE_PATH = RUNTIME_DIR / "state.json"
EVENTS_PATH = RUNTIME_DIR / "events.jsonl"
STOP_PATH = RUNTIME_DIR / "stop.requested"
D4_OPENING_PREFIX_MOVES = 20
D4_OPENING_PREFIX_DEPTHS = (4, 8, 12, D4_OPENING_PREFIX_MOVES)


class GpuBusyError(RuntimeError):
    """Raised when an exclusive GPU preflight finds compute processes."""


def validate_arena_config(config: dict[str, Any]) -> None:
    """Reject arena settings that can silently collapse independent games."""
    arena = config["arena"]
    games = int(arena["games"])
    candidate_as_black = int(arena["candidate_as_black"])
    candidate_as_white = int(arena["candidate_as_white"])
    if games <= 0 or candidate_as_black < 0 or candidate_as_white < 0:
        raise ValueError("arena games and color quotas must be non-negative")
    if candidate_as_black + candidate_as_white != games:
        raise ValueError("arena color quotas must sum to arena games")
    threshold = float(arena["promotion_threshold"])
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("arena promotion_threshold must be within (0, 1]")
    draw_score = float(arena["draw_score"])
    if not np.isfinite(draw_score) or not 0.0 <= draw_score <= 1.0:
        raise ValueError("arena draw_score must be within [0, 1]")
    if bool(arena.get("early_terminate_irreversible", False)) and (
        not bool(arena.get("persistent_workers", False))
        or int(arena.get("games_per_process", 1)) != 1
    ):
        raise ValueError(
            "arena irreversible early termination requires persistent "
            "single-game dispatch"
        )
    temperature = float(config["arena"]["action_temperature"])
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("arena action_temperature must be finite and greater than zero")
    early_temperature = float(
        config["arena"].get("chosen_move_temperature_early", temperature)
    )
    temperature_halflife = float(
        config["arena"].get("chosen_move_temperature_halflife", 19.0)
    )
    late_temperature = float(
        config["arena"].get("chosen_move_temperature", temperature)
    )
    if not all(
        np.isfinite(value)
        for value in (early_temperature, temperature_halflife, late_temperature)
    ):
        raise ValueError("arena chosen move temperatures and halflife must be finite")
    if early_temperature <= 0.0 or late_temperature <= 0.0:
        raise ValueError("arena chosen move temperatures must be positive")
    if early_temperature < late_temperature:
        raise ValueError(
            "arena early chosen move temperature must not be below the late value"
        )
    if temperature_halflife <= 0.0:
        raise ValueError("arena chosen move temperature halflife must be positive")
    if bool(config["arena"]["paired_seeds"]):
        raise ValueError("arena paired_seeds must be false")
    unique_rate = float(config["health"]["arena_min_unique_trajectory_rate"])
    if not np.isfinite(unique_rate) or not 0.5 <= unique_rate <= 1.0:
        raise ValueError("arena minimum unique trajectory rate must be within [0.5, 1.0]")
    d4_rate = float(config["health"]["arena_min_d4_opening_unique_rate"])
    if not np.isfinite(d4_rate) or not 0.0 <= d4_rate <= 1.0:
        raise ValueError("arena minimum D4 opening unique rate must be within [0, 1]")


def validate_pcr_config(config: dict[str, Any]) -> None:
    simulations = [int(value) for value in config["mcts"]["pcr_sims"]]
    probabilities = [float(value) for value in config["mcts"]["pcr_probs"]]
    if len(simulations) != 2 or len(simulations) != len(probabilities):
        raise ValueError("PCR requires exactly one fast and one full search budget")
    if simulations != sorted(simulations) or simulations[0] <= 0:
        raise ValueError("PCR simulations must be positive and ordered fast-to-full")
    if not np.isclose(sum(probabilities), 1.0) or any(value <= 0.0 for value in probabilities):
        raise ValueError("PCR probabilities must be positive and sum to one")
    early_temperature = float(config["mcts"]["chosen_move_temperature_early"])
    temperature_halflife = float(config["mcts"]["chosen_move_temperature_halflife"])
    late_temperature = float(config["mcts"]["chosen_move_temperature"])
    if not all(
        np.isfinite(value)
        for value in (early_temperature, temperature_halflife, late_temperature)
    ):
        raise ValueError("chosen move temperatures and halflife must be finite")
    if early_temperature <= 0.0 or late_temperature <= 0.0:
        raise ValueError("chosen move temperatures must be positive")
    if early_temperature < late_temperature:
        raise ValueError("early chosen move temperature must not be below the late value")
    if temperature_halflife <= 0.0:
        raise ValueError("chosen move temperature halflife must be positive")


def validate_training_config(config: dict[str, Any]) -> None:
    training = config["training"]
    dataset_storage = str(config["dataset"].get("storage", "legacy_in_memory"))
    if dataset_storage not in ("legacy_in_memory", "packed_mmap"):
        raise ValueError(
            "dataset storage must be 'legacy_in_memory' or 'packed_mmap'"
        )
    raw_gpu_ids = training.get("gpu_ids")
    gpu_ids = (
        [int(value) for value in raw_gpu_ids]
        if raw_gpu_ids is not None
        else [int(training["gpu_id"])]
    )
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(value < 0 for value in gpu_ids):
        raise ValueError("training gpu_ids must be non-empty, unique, and non-negative")
    batch_size = int(training["batch_size"])
    if batch_size < 1 or batch_size % len(gpu_ids) != 0:
        raise ValueError("training batch_size must be positive and divisible by gpu count")
    if int(training["dataloader_workers"]) < 0:
        raise ValueError("training dataloader_workers must be non-negative")
    if int(training.get("omp_threads_per_rank", 4)) < 1:
        raise ValueError("training omp_threads_per_rank must be positive")
    ddp_bucket_cap_mb = float(training.get("ddp_bucket_cap_mb", 25.0))
    if not np.isfinite(ddp_bucket_cap_mb) or ddp_bucket_cap_mb <= 0.0:
        raise ValueError("training ddp_bucket_cap_mb must be finite and positive")
    if int(training["progress_interval_steps"]) < 1:
        raise ValueError("training progress_interval_steps must be positive")
    max_steps = int(training.get("max_steps_per_iteration", 0))
    if max_steps < int(training["minimum_steps"]):
        raise ValueError(
            "training max_steps_per_iteration must be at least minimum_steps"
        )


def validate_gpu_stage_config(config: dict[str, Any]) -> None:
    for stage_name in ("collection", "arena"):
        stage = config[stage_name]
        if int(stage["processes_per_gpu"]) < 1:
            raise ValueError(f"{stage_name} processes_per_gpu must be positive")
        if int(stage.get("games_per_process", 1)) < 1:
            raise ValueError(f"{stage_name} games_per_process must be positive")
        if (
            bool(stage.get("persistent_workers", False))
            and int(stage.get("games_per_process", 1)) != 1
        ):
            raise ValueError(
                f"{stage_name} persistent workers require single-game dispatch"
            )


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    validate_arena_config(config)
    validate_pcr_config(config)
    validate_training_config(config)
    validate_gpu_stage_config(config)
    model = config["model"]
    if model.get("class") != "SignedKomiGoResNet":
        raise ValueError("model must preserve the warm-start signed-komi input")
    if model.get("input_features") != (
        "empty_self_opponent_plus_current_player_signed_komi"
    ):
        raise ValueError("model input feature contract changed")
    if model.get("signed_komi_scale") != "komi_over_board_area":
        raise ValueError("signed komi must be normalized by board area")
    initial = config.get("initial_model")
    if not isinstance(initial, dict):
        raise ValueError("initial_model warm-start provenance is required")
    source_sha256 = str(initial.get("source_sha256", ""))
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("initial_model.source_sha256 must be lowercase SHA-256")
    return config


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_runtime_dirs() -> None:
    for path in (RUNTIME_DIR, LOGS_DIR, MANIFEST_DIR, EXP_DATA_ROOT, CHECKPOINT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def _fsync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(event: str, **payload: Any) -> None:
    ensure_runtime_dirs()
    record = {"timestamp": utc_timestamp(), "event": event, **payload}
    with open(EVENTS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_data_path(path: Path) -> str:
    return str(path.resolve().relative_to(GAME_DATA_ROOT))


def stop_requested(path: Path = STOP_PATH) -> bool:
    return path.exists()


def canonical_d4_prefix(
    moves: np.ndarray,
    board_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return the lexicographically smallest D4 transform of the opening."""
    variants: list[tuple[tuple[int, int], ...]] = []
    prefix = [
        (int(row), int(col)) for row, col in moves[:D4_OPENING_PREFIX_MOVES]
    ]
    for transform in range(8):
        transformed: list[tuple[int, int]] = []
        for original_row, original_col in prefix:
            row, col = original_row, original_col
            if row >= 0 and col >= 0:
                if transform >= 4:
                    col = board_size - 1 - col
                for _ in range(transform % 4):
                    row, col = col, board_size - 1 - row
            transformed.append((row, col))
        variants.append(tuple(transformed))
    return min(variants)


def validate_npz(
    path: Path,
    *,
    board_size: int,
    require_mcts: bool,
    allowed_visit_totals: Iterable[int] = (100, 600),
) -> dict[str, Any]:
    """Validate one immutable game file and return compact statistics."""
    allowed = {int(x) for x in allowed_visit_totals}
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"boards", "moves", "winner", "num_moves", "board_size", "termination"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"missing fields {missing}")

            n_moves = int(data["num_moves"])
            boards = data["boards"]
            moves = data["moves"]
            if int(data["board_size"]) != board_size:
                raise ValueError(f"board_size={int(data['board_size'])}, expected {board_size}")
            if boards.shape != (n_moves, board_size, board_size):
                raise ValueError(f"boards shape {boards.shape}, expected {(n_moves, board_size, board_size)}")
            if moves.shape != (n_moves, 2):
                raise ValueError(f"moves shape {moves.shape}, expected {(n_moves, 2)}")
            winner = int(data["winner"])
            if winner not in (0, 1, 2):
                raise ValueError(f"invalid winner {winner}")

            stats: dict[str, Any] = {
                "positions": n_moves,
                "winner": winner,
                "termination": str(data["termination"]),
                "early_passes": int(np.all(moves[: min(n_moves, 10)] < 0, axis=1).sum()),
                "early_positions": min(n_moves, 10),
            }

            if require_mcts:
                mcts_required = {
                    "mcts_visits",
                    "mcts_temperatures",
                    "mcts_root_values",
                    "mcts_policy_priors",
                    "mcts_q_values",
                    "is_teacher",
                }
                missing_mcts = sorted(mcts_required - set(data.files))
                if missing_mcts:
                    raise ValueError(f"missing MCTS fields {missing_mcts}")
                actions = board_size * board_size + 1
                visits = data["mcts_visits"]
                teachers = data["is_teacher"]
                temps = data["mcts_temperatures"]
                priors = data["mcts_policy_priors"]
                roots = data["mcts_root_values"]
                q_values = data["mcts_q_values"]
                if visits.shape != (n_moves, actions):
                    raise ValueError(f"mcts_visits shape {visits.shape}, expected {(n_moves, actions)}")
                if teachers.shape != (n_moves,):
                    raise ValueError(f"is_teacher shape {teachers.shape}, expected {(n_moves,)}")
                totals = visits.sum(axis=1).astype(np.int64)
                actual = {int(x) for x in np.unique(totals)}
                if not actual or not actual.issubset(allowed):
                    raise ValueError(f"visit totals {sorted(actual)} not in {sorted(allowed)}")
                expected_teachers = totals == max(allowed)
                if not np.array_equal(np.asarray(teachers, dtype=np.bool_), expected_teachers):
                    raise ValueError("is_teacher must be true exactly for full-search positions")
                for name, array in (
                    ("mcts_temperatures", temps),
                    ("mcts_policy_priors", priors),
                    ("mcts_root_values", roots),
                    ("mcts_q_values", q_values),
                ):
                    if not np.isfinite(array).all():
                        raise ValueError(f"{name} contains NaN/Inf")
                if priors.shape != (n_moves, actions):
                    raise ValueError(f"mcts_policy_priors shape {priors.shape}")
                prior_sums = priors.sum(axis=1)
                if not np.allclose(prior_sums, 1.0, atol=2e-4):
                    raise ValueError("MCTS policy priors do not sum to one")
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    policies = np.zeros_like(visits, dtype=np.float32)
                    for i in range(n_moves):
                        tau = float(temps[i])
                        if tau == 0.0:
                            policies[i, int(np.argmax(visits[i]))] = 1.0
                        else:
                            powered = np.power(visits[i].astype(np.float32), 1.0 / tau)
                            denom = float(powered.sum())
                            if denom <= 0:
                                raise ValueError(f"zero policy total at row {i}")
                            policies[i] = powered / denom
                if not np.isfinite(policies).all() or not np.allclose(policies.sum(axis=1), 1.0, atol=1e-5):
                    raise ValueError("normalized MCTS policies are invalid")
                unique, counts = np.unique(totals, return_counts=True)
                stats["visit_budgets"] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
                stats["teacher_positions"] = int(np.asarray(teachers).sum())
                stats["value_only_positions"] = int(n_moves - np.asarray(teachers).sum())
            return stats
    except Exception as exc:
        raise ValueError(f"invalid NPZ {path}: {exc}") from exc


def validate_data_dir(
    path: Path,
    *,
    expected_games: int,
    board_size: int,
    require_mcts: bool,
    allowed_visit_totals: Iterable[int] = (100, 600),
) -> dict[str, Any]:
    files = sorted(path.glob("*.npz")) if path.exists() else []
    if len(files) != expected_games:
        raise ValueError(f"{path}: found {len(files)} games, expected {expected_games}")
    aggregate = {
        "games": len(files),
        "positions": 0,
        "teacher_positions": 0,
        "value_only_positions": 0,
        "early_passes": 0,
        "early_positions": 0,
        "terminations": {},
        "visit_budgets": {},
    }
    for file_path in files:
        item = validate_npz(
            file_path,
            board_size=board_size,
            require_mcts=require_mcts,
            allowed_visit_totals=allowed_visit_totals,
        )
        aggregate["positions"] += item["positions"]
        aggregate["teacher_positions"] += item.get("teacher_positions", 0)
        aggregate["value_only_positions"] += item.get("value_only_positions", 0)
        aggregate["early_passes"] += item["early_passes"]
        aggregate["early_positions"] += item["early_positions"]
        term = item["termination"]
        aggregate["terminations"][term] = aggregate["terminations"].get(term, 0) + 1
        for budget, count in item.get("visit_budgets", {}).items():
            aggregate["visit_budgets"][budget] = aggregate["visit_budgets"].get(budget, 0) + count
    return aggregate


def publish_directory(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to replace published directory {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    _fsync_parent(final)


def disk_free_gib(path: Path = REPO_ROOT) -> float:
    return shutil.disk_usage(path).free / 1024**3


def gpu_inventory() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU line: {line}")
        out.append({
            "index": int(parts[0]),
            "uuid": parts[1],
            "name": parts[2],
            "memory_total_mib": int(parts[3]),
            "memory_used_mib": int(parts[4]),
            "utilization_percent": int(parts[5]),
        })
    return out


def gpu_compute_processes() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        owner_query = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        out.append({
            "gpu_uuid": parts[0],
            "pid": pid,
            "process_name": parts[2],
            "used_memory_mib": None if parts[3] == "N/A" else int(parts[3]),
            "owner": owner_query.stdout.strip() or "unknown",
        })
    return out


def preflight_gpus(gpu_ids: Iterable[int], *, require_idle: bool = False) -> dict[str, Any]:
    selected = {int(x) for x in gpu_ids}
    inventory = gpu_inventory()
    by_index = {gpu["index"]: gpu for gpu in inventory}
    missing = sorted(selected - set(by_index))
    if missing:
        raise RuntimeError(f"requested GPU IDs do not exist: {missing}")
    uuid_to_index = {gpu["uuid"]: gpu["index"] for gpu in inventory}
    processes = []
    for process in gpu_compute_processes():
        index = uuid_to_index.get(process["gpu_uuid"])
        if index in selected:
            processes.append({**process, "gpu_index": index})
    if processes and require_idle:
        details = "; ".join(
            f"GPU {p['gpu_index']} pid={p['pid']} owner={p['owner']} cmd={p['process_name']}"
            for p in processes
        )
        raise GpuBusyError(f"selected GPUs are not idle: {details}")
    return {
        "selected": [by_index[index] for index in sorted(selected)],
        "compute_processes": processes,
        "require_idle": require_idle,
    }
