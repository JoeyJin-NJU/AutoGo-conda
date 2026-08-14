#!/usr/bin/env python
"""Incremental mmap cache with sample-exact DrawAwareGoDataset semantics."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from color_model import signed_komi_for_current_player
from torch.utils.data import Dataset

BLACK = 1
WHITE = 2
CACHE_SCHEMA_VERSION = 2


def _record_dtype(board_size: int) -> np.dtype:
    actions = board_size * board_size + 1
    return np.dtype(
        [
            ("board", np.int8, (board_size, board_size)),
            ("move", np.int16, (2,)),
            ("winner", np.int8),
            ("is_white", np.bool_),
            ("komi", np.float64),
            ("has_mcts", np.bool_),
            ("mcts_visits", np.int32, (actions,)),
            ("mcts_temperature", np.float64),
            ("mcts_root_value", np.float64),
            ("is_teacher", np.bool_),
        ],
        align=False,
    )


def _cache_paths(data_dir: Path, cache_root: Path) -> tuple[Path, Path]:
    key = hashlib.sha256(str(data_dir.resolve()).encode()).hexdigest()
    return cache_root / f"{key}.npy", cache_root / f"{key}.json"


def _load_source_snapshot(data_dir: Path) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    data_dir = data_dir.resolve()
    index_path = data_dir / "index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid dataset index: {index_path}") from exc
    if not isinstance(index, dict) or not index:
        raise ValueError(f"dataset index must be a non-empty object: {index_path}")

    indexed_files: list[tuple[str, int]] = []
    file_stats: list[list[int | str]] = []
    for name, count_value in index.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
            raise ValueError(f"unsafe NPZ name in {index_path}: {name!r}")
        count = int(count_value)
        if count < 0:
            raise ValueError(f"negative position count for {name} in {index_path}")
        source = data_dir / name
        stat = source.stat()
        indexed_files.append((name, count))
        file_stats.append([name, count, int(stat.st_size), int(stat.st_mtime_ns)])

    actual_names = {path.name for path in data_dir.glob("*.npz")}
    if actual_names != {name for name, _ in indexed_files}:
        raise ValueError(f"dataset index does not match NPZ files: {index_path}")

    fingerprint_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_dir": str(data_dir),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "files": file_stats,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot = {
        "source_dir": str(data_dir),
        "source_fingerprint": fingerprint,
        "num_files": len(indexed_files),
        "positions": sum(count for _, count in indexed_files),
    }
    return indexed_files, snapshot


def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_is_reusable(
    data_path: Path,
    metadata_path: Path,
    *,
    snapshot: dict[str, Any],
    board_size: int,
) -> bool:
    metadata = _load_metadata(metadata_path)
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_dir": snapshot["source_dir"],
        "source_fingerprint": snapshot["source_fingerprint"],
        "board_size": board_size,
        "num_files": snapshot["num_files"],
        "positions": snapshot["positions"],
        "record_itemsize": _record_dtype(board_size).itemsize,
    }
    if metadata is None or any(metadata.get(key) != value for key, value in expected.items()):
        return False
    try:
        records = np.load(data_path, mmap_mode="r", allow_pickle=False)
        valid = records.shape == (snapshot["positions"],) and records.dtype == _record_dtype(
            board_size
        )
        del records
        return bool(valid)
    except (OSError, ValueError, TypeError):
        return False


def _checked_integer_array(value: np.ndarray, dtype: np.dtype, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must have integer dtype, got {array.dtype}")
    if array.size:
        limits = np.iinfo(dtype)
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < limits.min or maximum > limits.max:
            raise ValueError(
                f"{label} range [{minimum}, {maximum}] cannot be represented by {dtype}"
            )
    return array.astype(dtype, copy=False)


def _build_cache_file(
    data_dir: Path,
    indexed_files: list[tuple[str, int]],
    snapshot: dict[str, Any],
    *,
    data_path: Path,
    metadata_path: Path,
    board_size: int,
) -> None:
    dtype = _record_dtype(board_size)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporary_data = data_path.with_name(f".{data_path.name}.{token}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    records: np.memmap | None = None
    try:
        records = np.lib.format.open_memmap(
            temporary_data,
            mode="w+",
            dtype=dtype,
            shape=(int(snapshot["positions"]),),
        )
        records["has_mcts"] = False
        records["komi"] = np.nan
        records["mcts_visits"] = 0
        records["mcts_temperature"] = 0.0
        records["mcts_root_value"] = 0.0
        records["is_teacher"] = False

        offset = 0
        actions = board_size * board_size + 1
        for name, expected_count in indexed_files:
            source = data_dir / name
            with np.load(source, allow_pickle=False) as data:
                count = int(data["num_moves"])
                if count != expected_count:
                    raise ValueError(
                        f"index count {expected_count} != num_moves {count}: {source}"
                    )
                stop = offset + count
                target = records[offset:stop]
                boards = np.asarray(data["boards"])
                moves = np.asarray(data["moves"])
                if boards.shape != (count, board_size, board_size):
                    raise ValueError(f"invalid boards shape {boards.shape}: {source}")
                if moves.shape != (count, 2):
                    raise ValueError(f"invalid moves shape {moves.shape}: {source}")
                target["board"] = _checked_integer_array(
                    boards, np.dtype(np.int8), label=f"boards in {source}"
                )
                target["move"] = _checked_integer_array(
                    moves, np.dtype(np.int16), label=f"moves in {source}"
                )
                winner = int(data["winner"])
                if winner not in (0, BLACK, WHITE):
                    raise ValueError(f"invalid winner {winner}: {source}")
                target["winner"] = winner
                target["is_white"] = np.arange(count) % 2 == 1
                if "komi" in data:
                    stored_komi = np.asarray(data["komi"])
                    if stored_komi.size != 1:
                        raise ValueError(
                            f"invalid komi shape {stored_komi.shape}: {source}"
                        )
                    sample_komi = float(stored_komi.item())
                    if not np.isfinite(sample_komi):
                        raise ValueError(f"non-finite komi {sample_komi}: {source}")
                    target["komi"] = sample_komi

                if "mcts_visits" in data:
                    visits = np.asarray(data["mcts_visits"])
                    if visits.shape != (count, actions):
                        raise ValueError(f"invalid mcts_visits shape {visits.shape}: {source}")
                    temperatures = np.asarray(data["mcts_temperatures"])
                    root_values = np.asarray(data["mcts_root_values"])
                    if temperatures.shape != (count,):
                        raise ValueError(
                            f"invalid mcts_temperatures shape {temperatures.shape}: {source}"
                        )
                    if root_values.shape != (count,):
                        raise ValueError(
                            f"invalid mcts_root_values shape {root_values.shape}: {source}"
                        )
                    target["has_mcts"] = True
                    target["mcts_visits"] = _checked_integer_array(
                        visits, np.dtype(np.int32), label=f"mcts_visits in {source}"
                    )
                    target["mcts_temperature"] = temperatures.astype(np.float64)
                    target["mcts_root_value"] = root_values.astype(np.float64)

                if "is_teacher" in data:
                    teacher = np.asarray(data["is_teacher"])
                    if teacher.shape != (count,):
                        raise ValueError(f"invalid is_teacher shape {teacher.shape}: {source}")
                    target["is_teacher"] = teacher.astype(np.bool_)
                offset = stop

        if offset != snapshot["positions"]:
            raise ValueError(
                f"packed {offset} positions but expected {snapshot['positions']}: {data_dir}"
            )
        records.flush()
        del records
        records = None
        with temporary_data.open("rb") as handle:
            os.fsync(handle.fileno())

        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_dir": snapshot["source_dir"],
            "source_fingerprint": snapshot["source_fingerprint"],
            "board_size": board_size,
            "num_files": snapshot["num_files"],
            "positions": snapshot["positions"],
            "record_itemsize": dtype.itemsize,
        }
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        with temporary_metadata.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_data, data_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        if records is not None:
            del records
        temporary_data.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def prepare_packed_cache(
    data_dirs: list[str | Path],
    *,
    cache_root: str | Path,
    board_size: int,
) -> dict[str, int | float]:
    """Build missing per-directory caches and reuse validated immutable ones."""
    started = time.perf_counter()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    built_dirs = 0
    reused_dirs = 0
    positions = 0
    for value in data_dirs:
        data_dir = Path(value).resolve()
        indexed_files, snapshot = _load_source_snapshot(data_dir)
        positions += int(snapshot["positions"])
        data_path, metadata_path = _cache_paths(data_dir, cache_root)
        if _cache_is_reusable(
            data_path,
            metadata_path,
            snapshot=snapshot,
            board_size=board_size,
        ):
            reused_dirs += 1
            continue
        _build_cache_file(
            data_dir,
            indexed_files,
            snapshot,
            data_path=data_path,
            metadata_path=metadata_path,
            board_size=board_size,
        )
        built_dirs += 1
    return {
        "built_dirs": built_dirs,
        "reused_dirs": reused_dirs,
        "positions": positions,
        "elapsed_seconds": time.perf_counter() - started,
    }


class PackedDrawAwareGoDataset(Dataset):
    """Read the packed cache with DrawAwareGoDataset-equivalent outputs."""

    def __init__(
        self,
        data_dirs: list[str | Path],
        *,
        cache_root: str | Path,
        board_size: int,
        komi: float,
        load_mcts_policy: bool,
        load_is_teacher: bool,
        policy_target_temperature: float | None,
    ) -> None:
        if policy_target_temperature is not None:
            policy_target_temperature = float(policy_target_temperature)
            if not np.isfinite(policy_target_temperature) or policy_target_temperature < 0:
                raise ValueError("policy_target_temperature must be finite and non-negative")
        self.data_dirs = [Path(value).resolve() for value in data_dirs]
        self.cache_root = Path(cache_root).resolve()
        self.board_size = int(board_size)
        self.komi = float(komi)
        self.load_mcts_policy = bool(load_mcts_policy)
        self.load_is_teacher = bool(load_is_teacher)
        self.policy_target_temperature = policy_target_temperature
        self._arrays: list[np.memmap] = []
        self._num_files = 0
        totals: list[int] = []
        expected_dtype = _record_dtype(self.board_size)
        for data_dir in self.data_dirs:
            data_path, metadata_path = _cache_paths(data_dir, self.cache_root)
            metadata = _load_metadata(metadata_path)
            if metadata is None or metadata.get("source_dir") != str(data_dir):
                raise ValueError(f"missing or invalid packed metadata for {data_dir}")
            records = np.load(data_path, mmap_mode="r", allow_pickle=False)
            if records.ndim != 1 or records.dtype != expected_dtype:
                raise ValueError(f"invalid packed data for {data_dir}")
            if len(records) != int(metadata.get("positions", -1)):
                raise ValueError(f"packed position count mismatch for {data_dir}")
            self._arrays.append(records)
            totals.append(len(records))
            self._num_files += int(metadata["num_files"])
        self.total_positions = sum(totals)
        self._dir_cumsum = np.cumsum([0] + totals)

    def __len__(self) -> int:
        return self.total_positions

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.total_positions:
            raise IndexError(f"Index {idx} out of range [0, {self.total_positions})")
        dir_idx = int(np.searchsorted(self._dir_cumsum[1:], idx, side="right"))
        local_idx = idx - int(self._dir_cumsum[dir_idx])
        record = self._arrays[dir_idx][local_idx]

        board = record["board"].copy()
        move = record["move"]
        raw_winner = int(record["winner"])
        is_white = bool(record["is_white"])
        current_player = WHITE if is_white else BLACK
        is_expert = raw_winner == current_player
        if is_white:
            board = np.where(
                board == BLACK,
                WHITE,
                np.where(board == WHITE, BLACK, board),
            )
        current_wins = 1 if raw_winner == current_player else 0
        value_target = 0.5 if raw_winner == 0 else float(current_wins)
        sample_komi = float(record["komi"])
        if np.isnan(sample_komi):
            sample_komi = self.komi
        result: dict[str, Any] = {
            "board": torch.from_numpy(board).float(),
            "is_expert": is_expert,
            "move": torch.tensor([move[0], move[1]], dtype=torch.long),
            "winner": torch.tensor(value_target, dtype=torch.float32),
            "signed_komi": signed_komi_for_current_player(
                is_white,
                komi=sample_komi,
                board_size=self.board_size,
            ),
        }

        if self.load_mcts_policy:
            actions = self.board_size * self.board_size + 1
            if bool(record["has_mcts"]):
                visits = record["mcts_visits"].astype(np.float32)
                temperature = self.policy_target_temperature
                if temperature is None:
                    temperature = float(record["mcts_temperature"])
                root_value = float(record["mcts_root_value"])
                if temperature == 0:
                    mcts_policy = np.zeros(actions, dtype=np.float32)
                    if visits.sum() > 0:
                        mcts_policy[np.argmax(visits)] = 1.0
                else:
                    visits_temp = np.power(visits, 1.0 / temperature)
                    total = visits_temp.sum()
                    mcts_policy = (
                        visits_temp / total
                        if total > 0
                        else np.zeros(actions, dtype=np.float32)
                    )
                has_mcts = True
            else:
                smooth_eps = 0.1
                mcts_policy = np.full(actions, smooth_eps / actions, dtype=np.float32)
                row, col = int(move[0]), int(move[1])
                target_idx = actions - 1 if row < 0 else row * self.board_size + col
                mcts_policy[target_idx] += 1.0 - smooth_eps
                root_value = float(current_wins)
                has_mcts = False
            result["mcts_policy"] = torch.from_numpy(mcts_policy).float()
            result["mcts_root_value"] = torch.tensor(root_value, dtype=torch.float32)
            result["has_mcts"] = has_mcts

        if self.load_is_teacher:
            result["is_teacher"] = torch.tensor(
                bool(record["is_teacher"]), dtype=torch.float32
            )
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_positions": self.total_positions,
            "num_dirs": len(self.data_dirs),
            "num_files": self._num_files,
            "board_size": self.board_size,
            "data_dirs": [str(path) for path in self.data_dirs],
        }
