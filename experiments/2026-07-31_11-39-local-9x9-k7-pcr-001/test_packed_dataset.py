from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from packed_dataset import PackedDrawAwareGoDataset, prepare_packed_cache
from train import DrawAwareGoDataset


def _write_game(
    path: Path,
    *,
    boards: np.ndarray,
    moves: np.ndarray,
    winner: int,
    komi: float | None = None,
    visits: np.ndarray | None = None,
    temperatures: np.ndarray | None = None,
    root_values: np.ndarray | None = None,
    is_teacher: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "boards": boards,
        "moves": moves,
        "winner": np.array(winner, dtype=np.int64),
        "num_moves": np.array(len(moves), dtype=np.int64),
        "board_size": np.array(boards.shape[-1], dtype=np.int64),
    }
    if komi is not None:
        payload["komi"] = np.array(komi, dtype=np.float64)
    if visits is not None:
        payload["mcts_visits"] = visits
        payload["mcts_temperatures"] = temperatures
        payload["mcts_root_values"] = root_values
    if is_teacher is not None:
        payload["is_teacher"] = is_teacher
    np.savez_compressed(path, **payload)


def _make_data(data_root: Path) -> list[Path]:
    first = data_root / "first"
    second = data_root / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    actions = 10
    boards_a = np.zeros((3, 3, 3), dtype=np.int8)
    boards_a[1, 0, 0] = 1
    boards_a[1, 0, 1] = 2
    moves_a = np.array([[0, 0], [1, 1], [-1, -1]], dtype=np.int16)
    visits_a = np.zeros((3, actions), dtype=np.int16)
    visits_a[0, :3] = [1, 2, 3]
    visits_a[1, :3] = [4, 0, 1]
    visits_a[2, -1] = 8
    _write_game(
        first / "game-b.npz",
        boards=boards_a,
        moves=moves_a,
        winner=0,
        komi=6.5,
        visits=visits_a,
        temperatures=np.array([1.0, 0.5, 0.0], dtype=np.float32),
        root_values=np.array([-0.25, 0.0, 0.75], dtype=np.float32),
        is_teacher=np.array([True, False, True], dtype=np.bool_),
    )

    boards_b = np.full((2, 3, 3), 2, dtype=np.int8)
    moves_b = np.array([[2, 2], [0, 2]], dtype=np.int16)
    _write_game(
        first / "game-a.npz",
        boards=boards_b,
        moves=moves_b,
        winner=2,
        komi=7.0,
    )
    (first / "index.json").write_text(
        json.dumps({"game-b.npz": 3, "game-a.npz": 2})
    )

    boards_c = np.arange(18, dtype=np.int8).reshape(2, 3, 3) % 3
    moves_c = np.array([[2, 0], [2, 1]], dtype=np.int16)
    visits_c = np.zeros((2, actions), dtype=np.int16)
    visits_c[:, 4] = [16, 32]
    _write_game(
        second / "game-c.npz",
        boards=boards_c,
        moves=moves_c,
        winner=1,
        komi=7.5,
        visits=visits_c,
        temperatures=np.array([0.3, 1.0], dtype=np.float32),
        root_values=np.array([0.125, -0.5], dtype=np.float32),
        is_teacher=np.array([False, True], dtype=np.bool_),
    )
    (second / "index.json").write_text(json.dumps({"game-c.npz": 2}))
    return [first, second]


def _assert_samples_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert left[key].dtype == right[key].dtype, key
            assert torch.equal(left[key], right[key]), key
        else:
            assert type(left[key]) is type(right[key]), key
            assert left[key] == right[key], key


def test_packed_dataset_is_sample_exact_for_fixed_and_saved_temperatures(
    tmp_path: Path,
) -> None:
    data_dirs = _make_data(tmp_path / "data")
    cache_root = tmp_path / "cache"
    summary = prepare_packed_cache(data_dirs, cache_root=cache_root, board_size=3)
    assert summary["built_dirs"] == 2
    assert summary["reused_dirs"] == 0
    assert summary["positions"] == 7

    for target_temperature in (1.0, None):
        legacy = DrawAwareGoDataset(
            data_dirs,
            komi=5.5,
            load_mcts_policy=True,
            load_is_teacher=True,
            in_memory=True,
            policy_target_temperature=target_temperature,
        )
        packed = PackedDrawAwareGoDataset(
            data_dirs,
            cache_root=cache_root,
            board_size=3,
            komi=5.5,
            load_mcts_policy=True,
            load_is_teacher=True,
            policy_target_temperature=target_temperature,
        )
        assert len(packed) == len(legacy) == 7
        assert packed.get_stats()["num_files"] == legacy.get_stats()["num_files"] == 3
        for index in range(len(legacy)):
            _assert_samples_equal(legacy[index], packed[index])
        np.testing.assert_allclose(
            [packed[index]["signed_komi"].item() for index in range(len(packed))],
            [-6.5 / 9, 6.5 / 9, -6.5 / 9, -7.0 / 9, 7.0 / 9, -7.5 / 9, 7.5 / 9],
        )


def test_packed_cache_reuses_unchanged_data_and_rebuilds_changed_source(
    tmp_path: Path,
) -> None:
    data_dirs = _make_data(tmp_path / "data")
    cache_root = tmp_path / "cache"
    first = prepare_packed_cache(data_dirs, cache_root=cache_root, board_size=3)
    second = prepare_packed_cache(data_dirs, cache_root=cache_root, board_size=3)
    assert first["built_dirs"] == 2
    assert second["built_dirs"] == 0
    assert second["reused_dirs"] == 2

    source = data_dirs[1] / "game-c.npz"
    with np.load(source, allow_pickle=False) as original:
        payload = {key: value.copy() for key, value in original.items()}
    payload["boards"][0, 0, 0] = 2
    np.savez_compressed(source, **payload)

    rebuilt = prepare_packed_cache(data_dirs, cache_root=cache_root, board_size=3)
    assert rebuilt["built_dirs"] == 1
    assert rebuilt["reused_dirs"] == 1

    legacy = DrawAwareGoDataset(
        data_dirs,
        komi=7.0,
        load_mcts_policy=True,
        load_is_teacher=True,
        in_memory=True,
        policy_target_temperature=1.0,
    )
    packed = PackedDrawAwareGoDataset(
        data_dirs,
        cache_root=cache_root,
        board_size=3,
        komi=7.0,
        load_mcts_policy=True,
        load_is_teacher=True,
        policy_target_temperature=1.0,
    )
    for index in range(len(legacy)):
        _assert_samples_equal(legacy[index], packed[index])
