from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import common
import numpy as np
import pytest
import torch
from arena import (
    _read_side,
    evaluate_arena,
    gatekeeper_decision,
)
from bootstrap import discover_game_indices, extract_game_index
from common import (
    atomic_write_text,
    load_config,
    sha256_file,
    stop_requested,
    validate_arena_config,
    validate_gpu_stage_config,
    validate_npz,
    validate_training_config,
)
from controller import (
    Controller,
    _chunk_distribution,
    _chunk_missing_game_ranges,
    _gpu_process_slots,
    _interleave_arena_game_tasks,
    _split_count,
    _summarize_worker_error,
    _training_gpu_ids,
    build_run_games_command,
    run_dynamic_process_pool,
)
from run_games import (
    PersistentGameRunner,
    _game_invocations,
    _mark_full_search_teachers,
    _reset_reused_search_state,
    _use_root_noise,
    choose_pcr_budget,
    chosen_move_temperature,
    move_search_settings,
    persistent_worker_loop,
    temperature_schedule_config,
)
from status import (
    _arena_history,
    _arena_live_result,
    _gpu_status,
    _scan_npz_roots,
    render_dashboard,
)
from train import (
    DistributedContext,
    DrawAwareGoDataset,
    atomic_torch_save,
    build_policy_mask,
    ensure_dataset_indexes,
    make_loader,
)


def test_full_search_teacher_fast_value_only_and_bootstrap_fallback() -> None:
    batch = {
        "has_mcts": torch.tensor([True, True, False, False]),
        "is_teacher": torch.tensor([True, False, False, False]),
        "is_expert": torch.tensor([False, True, False, True]),
    }
    mask, fallback = build_policy_mask(batch)
    assert mask.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert fallback.tolist() == [False, False, False, True]


def test_root_noise_can_include_fast_searches() -> None:
    assert _use_root_noise(enabled=True, full_search=True, full_search_only=True)
    assert not _use_root_noise(enabled=True, full_search=False, full_search_only=True)
    assert _use_root_noise(enabled=True, full_search=False, full_search_only=False)
    assert not _use_root_noise(enabled=False, full_search=True, full_search_only=False)


def test_evaluation_loader_can_reuse_persistent_workers() -> None:
    dataset = [torch.tensor([0.0]), torch.tensor([1.0])]
    evaluation = make_loader(
        dataset,
        batch_size=1,
        workers=1,
        seed=7,
        epoch=0,
        shuffle=False,
    )
    training = make_loader(
        dataset,
        batch_size=1,
        workers=1,
        seed=7,
        epoch=0,
        shuffle=True,
    )
    persistent_evaluation = make_loader(
        dataset,
        batch_size=1,
        workers=1,
        seed=7,
        epoch=0,
        shuffle=False,
        persistent_workers=True,
    )

    assert evaluation.persistent_workers is False
    assert training.persistent_workers is True
    assert persistent_evaluation.persistent_workers is True


def test_distributed_loader_preserves_global_batch_step_count() -> None:
    dataset = list(range(24))
    loaders = [
        make_loader(
            dataset,
            batch_size=4,
            workers=0,
            seed=7,
            epoch=0,
            shuffle=False,
            rank=rank,
            world_size=3,
        )
        for rank in range(3)
    ]

    assert [len(loader) for loader in loaders] == [2, 2, 2]
    samples = [int(value) for loader in loaders for batch in loader for value in batch]
    assert sorted(samples) == list(range(24))


def test_distributed_context_properties() -> None:
    context = DistributedContext(rank=0, local_rank=0, world_size=3)
    assert context.enabled
    assert context.is_primary
    assert not DistributedContext(rank=1, local_rank=1, world_size=3).is_primary


def test_primary_rank_atomically_builds_dataset_index(tmp_path: Path) -> None:
    np.savez(tmp_path / "game000.npz", num_moves=np.array(3))
    np.savez(tmp_path / "game001.npz", num_moves=np.array(5))
    (tmp_path / "index.json").write_text("{broken")

    ensure_dataset_indexes(
        [tmp_path],
        DistributedContext(rank=0, local_rank=0, world_size=1),
    )

    assert (tmp_path / "index.json").read_text().endswith("\n")
    assert json.loads((tmp_path / "index.json").read_text()) == {
        "game000.npz": 3,
        "game001.npz": 5,
    }


def test_teacher_without_mcts_fails_closed() -> None:
    batch = {
        "has_mcts": torch.tensor([False]),
        "is_teacher": torch.tensor([True]),
        "is_expert": torch.tensor([True]),
    }
    with pytest.raises(ValueError, match="without MCTS"):
        build_policy_mask(batch)


def _write_mcts_npz(
    path: Path,
    teacher: bool = True,
    *,
    winner: int = 1,
    visits_total: int = 600,
) -> None:
    actions = 9 * 9 + 1
    visits = np.zeros((1, actions), dtype=np.int16)
    visits[0, 0] = visits_total
    priors = np.full((1, actions), 1.0 / actions, dtype=np.float32)
    np.savez_compressed(
        path,
        boards=np.zeros((1, 9, 9), dtype=np.int8),
        moves=np.array([[0, 0]], dtype=np.int16),
        winner=np.int64(winner),
        num_moves=np.int64(1),
        board_size=np.int64(9),
        termination=np.array("max_moves"),
        mcts_visits=visits,
        mcts_temperatures=np.array([0.3], dtype=np.float32),
        mcts_root_values=np.array([0.5], dtype=np.float32),
        mcts_policy_priors=priors,
        mcts_q_values=np.zeros((1, actions), dtype=np.float32),
        is_teacher=np.array([teacher], dtype=np.bool_),
    )


def test_npz_teacher_and_visit_validation(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    _write_mcts_npz(good, teacher=True)
    stats = validate_npz(good, board_size=9, require_mcts=True)
    assert stats["teacher_positions"] == 1
    fast = tmp_path / "fast.npz"
    _write_mcts_npz(fast, teacher=False, visits_total=100)
    stats = validate_npz(fast, board_size=9, require_mcts=True)
    assert stats["value_only_positions"] == 1
    bad = tmp_path / "bad.npz"
    _write_mcts_npz(bad, teacher=True, visits_total=100)
    with pytest.raises(ValueError, match="is_teacher"):
        validate_npz(bad, board_size=9, require_mcts=True)


def test_teacher_rewrite_uses_full_search_rows_only(tmp_path: Path) -> None:
    path = tmp_path / "mixed.npz"
    actions = 9 * 9 + 1
    visits = np.zeros((2, actions), dtype=np.int16)
    visits[:, 0] = [100, 600]
    np.savez_compressed(path, mcts_visits=visits, is_teacher=np.ones(2, dtype=np.bool_))

    _mark_full_search_teachers(path, full_simulations=600)

    with np.load(path, allow_pickle=False) as data:
        assert data["is_teacher"].tolist() == [False, True]


def test_pcr_configuration_and_sampling() -> None:
    config = load_config()
    assert config["mcts"]["pcr_sims"] == [1024, 2048]
    assert config["mcts"]["pcr_probs"] == [0.95, 0.05]
    samples = [choose_pcr_budget(seed, [1024, 2048], [0.95, 0.05]) for seed in range(4000)]
    assert set(samples) == {1024, 2048}
    assert samples.count(2048) / len(samples) == pytest.approx(0.05, abs=0.02)


def test_chosen_move_temperature_matches_katago_board_scaled_decay() -> None:
    kwargs = {"early": 0.75, "halflife": 19.0, "late": 0.15}

    assert chosen_move_temperature(0, 9, **kwargs) == pytest.approx(0.75)
    assert chosen_move_temperature(9, 9, **kwargs) == pytest.approx(0.45)
    assert chosen_move_temperature(18, 9, **kwargs) == pytest.approx(0.30)
    assert chosen_move_temperature(19, 19, **kwargs) == pytest.approx(0.45)


def test_arena_temperature_schedule_is_independent_from_collection() -> None:
    config = load_config()
    config["mcts"].update({
        "chosen_move_temperature_early": 0.75,
        "chosen_move_temperature_halflife": 19.0,
        "chosen_move_temperature": 0.15,
    })
    config["arena"].update({
        "chosen_move_temperature_early": 0.65,
        "chosen_move_temperature_halflife": 24.0,
        "chosen_move_temperature": 0.20,
    })

    assert temperature_schedule_config(config, is_arena=False) == (0.75, 19.0, 0.15)
    assert temperature_schedule_config(config, is_arena=True) == (0.65, 24.0, 0.20)


def test_arena_without_pcr_still_uses_dynamic_move_temperature() -> None:
    simulations, temperature, use_noise = move_search_settings(
        seed=7,
        turn_number=9,
        board_size=9,
        base_simulations=600,
        pcr_sims=None,
        pcr_probs=None,
        temperature_early=0.75,
        temperature_halflife=19.0,
        temperature_late=0.15,
        dirichlet_noise=False,
        dirichlet_full_search_only=False,
    )

    assert simulations == 600
    assert temperature == pytest.approx(0.45)
    assert use_noise is False


def test_leaf_batch_override_is_exposed_by_collector_cli() -> None:
    source = Path(__file__).with_name("run_games.py").read_text()
    assert 'parser.add_argument("--leaf-batch-size"' in source
    assert "leaf batch size must be positive" in source


def test_draw_value_target_is_neutral_and_not_policy_fallback(tmp_path: Path) -> None:
    game = tmp_path / "draw.npz"
    np.savez_compressed(
        game,
        boards=np.zeros((2, 9, 9), dtype=np.int8),
        moves=np.array([[0, 0], [0, 1]], dtype=np.int16),
        winner=np.int64(0),
        num_moves=np.int64(2),
        board_size=np.int64(9),
        termination=np.array("max_moves"),
    )
    dataset = DrawAwareGoDataset(
        tmp_path,
        komi=7.0,
        load_mcts_policy=True,
        load_is_teacher=True,
        in_memory=True,
    )
    sample = dataset[0]
    assert sample["winner"].item() == pytest.approx(0.5)
    assert sample["signed_komi"].item() == pytest.approx(-7.0 / 81.0)
    assert sample["is_expert"] is False
    mask, fallback = build_policy_mask({key: value.unsqueeze(0) if torch.is_tensor(value) else torch.tensor([value]) for key, value in sample.items() if key in {"has_mcts", "is_teacher", "is_expert"}})
    assert mask.item() == 0.0
    assert fallback.item() is False


def test_status_counts_staged_npz_instead_of_stale_state(tmp_path: Path) -> None:
    first = tmp_path / "matchup-a" / "shard000"
    second = tmp_path / "matchup-b" / "shard000"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_mcts_npz(first / "20260727-000000-game0000000.npz")
    _write_mcts_npz(second / "20260727-000001-game0000001.npz")

    progress = _scan_npz_roots([tmp_path])

    assert progress["games"] == 2
    assert progress["positions"] == 2
    assert [group["name"] for group in progress["groups"]] == ["matchup-a", "matchup-b"]


def test_status_groups_d4_equivalent_opening_prefixes(tmp_path: Path) -> None:
    group = tmp_path / "matchup-a" / "shard000"
    group.mkdir(parents=True)
    base = np.array([(index // 9, index % 9) for index in range(20)], dtype=np.int16)
    rotated = np.array([(col, 8 - row) for row, col in base], dtype=np.int16)
    different = base.copy()
    different[-1] = (8, 8)
    for index, moves in enumerate((base, rotated, different)):
        np.savez_compressed(
            group / f"game{index}.npz",
            moves=moves,
            num_moves=np.int64(len(moves)),
            winner=np.int64(1),
        )

    progress = _scan_npz_roots([tmp_path], board_size=9)
    result = progress["groups"][0]

    assert result["d4_opening_eligible"] == 3
    assert result["d4_opening_unique"] == 2
    assert result["d4_opening_unique_rate"] == pytest.approx(2 / 3)
    assert result["d4_opening_largest_cluster"] == 2
    assert result["d4_opening_largest_cluster_rate"] == pytest.approx(2 / 3)
    assert result["d4_opening_by_depth"]["4"] == {
        "eligible": 3,
        "unique": 1,
        "unique_rate": pytest.approx(1 / 3),
        "largest_cluster": 3,
        "largest_cluster_rate": pytest.approx(1.0),
    }
    assert result["d4_opening_by_depth"]["20"] == {
        "eligible": 3,
        "unique": 2,
        "unique_rate": pytest.approx(2 / 3),
        "largest_cluster": 2,
        "largest_cluster_rate": pytest.approx(2 / 3),
    }


def test_arena_persists_d4_opening_depth_profile(tmp_path: Path) -> None:
    root = tmp_path / "candidate-black"
    root.mkdir()
    base = np.array([(index // 9, index % 9) for index in range(20)], dtype=np.int16)
    rotated = np.array([(col, 8 - row) for row, col in base], dtype=np.int16)
    different = base.copy()
    different[-1] = (8, 8)
    for index, moves in enumerate((base, rotated, different)):
        np.savez_compressed(
            root / f"game{index}.npz",
            moves=moves,
            num_moves=np.int64(len(moves)),
            winner=np.int64(1),
            termination=np.array("max_moves"),
        )

    result = _read_side(root, 1, load_config())

    assert result["d4_opening_by_depth"]["4"]["unique"] == 1
    assert result["d4_opening_by_depth"]["4"]["largest_cluster"] == 3
    assert result["d4_opening_by_depth"]["20"]["unique"] == 2
    assert result["d4_opening_by_depth"]["20"]["largest_cluster"] == 2


def test_training_only_writes_iteration_final_checkpoint() -> None:
    config = load_config()
    assert "checkpoint_interval_steps" not in config["training"]
    assert "keep_recent_resumable" not in config["training"]
    assert "checkpointing" not in config

    source = Path(__file__).with_name("train.py").read_text()
    assert ".resume.pt" not in source
    assert 'parser.add_argument("--resume"' not in source
    assert 'f"iter{args.iteration:04d}-candidate.pt"' in source


def test_status_computes_live_arena_result_by_candidate_color(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black" / "shard000"
    white = tmp_path / "candidate-white" / "shard000"
    black.mkdir(parents=True)
    white.mkdir(parents=True)
    _write_mcts_npz(black / "black-win.npz", winner=1)
    _write_mcts_npz(black / "black-loss.npz", winner=2)
    _write_mcts_npz(white / "white-loss.npz", winner=1)
    _write_mcts_npz(white / "white-win.npz", winner=2)
    _write_mcts_npz(white / "draw.npz", winner=0)

    progress = _scan_npz_roots([tmp_path])
    result = _arena_live_result(
        progress["groups"], draw_score=0.5, promotion_threshold=0.55
    )

    assert result["games"] == 5
    assert (result["wins"], result["losses"], result["draws"]) == (2, 2, 1)
    assert result["score"] == pytest.approx(0.5)
    assert result["candidate_as_black"]["wins"] == 1
    assert result["candidate_as_white"]["wins"] == 1


def test_status_scan_ignores_inflight_game_files(tmp_path: Path) -> None:
    published = tmp_path / "candidate-black" / "shard000"
    inflight = published / ".inflight" / "pid-123-task"
    published.mkdir(parents=True)
    inflight.mkdir(parents=True)
    _write_mcts_npz(published / "complete.npz", winner=1)
    (inflight / "partial.npz").write_bytes(b"incomplete")

    progress = _scan_npz_roots([tmp_path])

    assert progress["games"] == 1
    assert progress["invalid"] == 0


def test_status_arena_history_ignores_early_stop_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    full_result = {
        "games": 182,
        "wins": 81,
        "losses": 101,
        "draws": 0,
        "score": 81 / 182,
        "promoted": False,
    }
    early_stop = {
        "iteration": 271,
        "games": 182,
        "candidate_points": 81.0,
        "decision": "reject",
        "remaining_games_skipped": 18,
    }
    (tmp_path / "arena-it0271.json").write_text(json.dumps(full_result))
    (tmp_path / "arena-it0271-early-stop.json").write_text(
        json.dumps(early_stop)
    )
    monkeypatch.setattr("status.RUNTIME_DIR", tmp_path)

    history = _arena_history()

    assert history == [{"iteration": 271, **full_result}]


def test_status_table_columns_share_fixed_boundaries() -> None:
    payload = {
        "experiment": "example",
        "generated_at": "2026-07-27T16:00:00+08:00",
        "experiment_age_seconds": 60,
        "controller_alive": True,
        "stage": "COLLECTING",
        "iteration": 8,
        "champion_iteration": 7,
        "controller_pid": 123,
        "child_pids": [456],
        "completed_games_current_stage": 1,
        "expected_games_current_stage": 10,
        "completed_positions_current_stage": 200,
        "stage_age_seconds": 30,
        "state_age_seconds": 1,
        "stop_requested": False,
        "last_error": None,
        "recovery_command": None,
        "current_data": {
            "name": "collect-it0008",
            "invalid": 0,
            "groups": [
                {"name": "as-black-vs-promoted-it0007", "games": 1, "positions": 200, "invalid": 0}
            ],
        },
        "latest_training": None,
        "arena_history": [{
            "iteration": 7,
            "wins": 65,
            "losses": 35,
            "draws": 0,
            "score": 0.65,
            "candidate_as_black": {"wins": 33, "losses": 17},
            "candidate_as_white": {"wins": 32, "losses": 18},
            "max_moves_rate": 0.04,
            "unique_trajectory_rate": 1.0,
            "promoted": True,
        }],
        "gpu_rows": [{
            "index": 3,
            "name": "NVIDIA RTX A6000",
            "utilization_percent": 94,
            "memory_used_mib": 3332,
            "memory_total_mib": 49140,
            "autogo_processes": 8,
            "other_processes": 0,
        }],
        "gpu_error": None,
        "total_npz_files": 100,
        "checkpoint_count": 2,
        "disk_free_gib": 1000.0,
        "newest_checkpoint": "/tmp/iter0007.pt",
        "newest_log": None,
        "newest_log_age_seconds": None,
        "newest_log_tail": [],
    }

    lines = render_dashboard(payload, 132).splitlines()
    table_pairs = [
        (next(line for line in lines if "Matchup / side" in line), next(line for line in lines if "as-black-vs" in line)),
        (next(line for line in lines if "W-L-D" in line), next(line for line in lines if "65-35-0" in line)),
        (next(line for line in lines if "Model" in line), next(line for line in lines if "NVIDIA RTX" in line)),
    ]
    for header, row in table_pairs:
        header_boundaries = [index for index, char in enumerate(header) if char == "|"]
        row_boundaries = [index for index, char in enumerate(row) if char == "|"]
        assert len(header_boundaries) >= 4
        assert row_boundaries == header_boundaries


def test_twelve_jobs_cover_seven_steady_state_matchups() -> None:
    distribution = _chunk_distribution(7, 12)
    assert distribution == [2, 2, 2, 2, 2, 1, 1]
    assert sum(distribution) == 12
    assert _split_count(50, 2) == [25, 25]


def test_missing_games_are_chunked_without_crossing_resume_gaps() -> None:
    assert _chunk_missing_game_ranges(10, {2, 6}, 3) == [
        (0, 2),
        (3, 3),
        (7, 3),
    ]
    assert _chunk_missing_game_ranges(4, set(), 2) == [(0, 2), (2, 2)]
    with pytest.raises(ValueError, match="games_per_process"):
        _chunk_missing_game_ranges(4, set(), 0)


def test_eight_mcts_processes_are_assigned_to_each_gpu() -> None:
    config = load_config()
    assert config["collection"]["processes_per_gpu"] == 8
    assert config["collection"]["games_per_process"] == 1
    assert config["collection"]["persistent_workers"] is True
    assert config["collection"]["workers_per_process"] == 1
    assert config["collection"]["num_jobs"] == 12
    assert config["collection"]["games_per_matchup"] == 50
    assert config["collection"]["selfplay_games"] == 50
    assert config["arena"]["processes_per_gpu"] == 8
    assert config["arena"]["games_per_process"] == 1
    assert config["arena"]["persistent_workers"] is True
    assert config["arena"]["workers_per_process"] == 1
    assert config["arena"]["games"] == 200
    assert config["arena"]["promotion_threshold"] == 0.5
    assert config["arena"]["early_terminate_irreversible"] is True
    assert config["mcts"]["leaf_batch_size"] == 32
    assert config["arena"]["leaf_batch_size"] == 32
    assert temperature_schedule_config(config, is_arena=False) == (0.75, 19.0, 0.15)
    assert temperature_schedule_config(config, is_arena=True) == (0.75, 19.0, 0.15)
    assert config["mcts"]["dirichlet_noise"] is True
    assert config["arena"]["dirichlet_noise"] is False
    assert config["health"]["arena_min_d4_opening_unique_rate"] == 0.0
    assert config["collection"]["gpu_ids"] == [2, 3, 4, 5]
    assert config["arena"]["gpu_ids"] == [2, 3, 4, 5]
    assert config["training"]["gpu_ids"] == [2, 3, 4, 5]
    assert _training_gpu_ids(config) == [2, 3, 4, 5]
    assert config["model"]["class"] == "SignedKomiGoResNet"
    assert config["model"]["parameters"] == 2_967_683
    assert config["training"]["batch_size"] % len(config["training"]["gpu_ids"]) == 0
    assert config["training"]["omp_threads_per_rank"] == 4
    assert config["training"]["ddp_bucket_cap_mb"] == 4.0
    assert config["training"]["progress_interval_steps"] == 1000
    assert config["training"]["max_steps_per_iteration"] == 6000
    assert config["health"]["require_idle_gpus"] is True
    slots = _gpu_process_slots([2, 3, 4, 5], processes_per_gpu=8)
    assert len(slots) == 32
    assert {gpu: slots.count(gpu) for gpu in (2, 3, 4, 5)} == {
        2: 8,
        3: 8,
        4: 8,
        5: 8,
    }


def test_status_includes_all_distributed_training_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    monkeypatch.setattr(
        "status.gpu_inventory",
        lambda: [
            {
                "index": gpu_id,
                "uuid": f"GPU-{gpu_id}",
                "name": "NVIDIA RTX A6000",
                "utilization_percent": 0,
                "memory_used_mib": 9,
                "memory_total_mib": 49140,
            }
            for gpu_id in range(8)
        ],
    )
    monkeypatch.setattr("status.gpu_compute_processes", lambda: [])

    rows, error = _gpu_status(config)

    assert error is None
    assert [row["index"] for row in rows] == [2, 3, 4, 5]


def test_gpu_preflight_allows_foreign_processes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "gpu_inventory",
        lambda: [{"index": 4, "uuid": "GPU-4"}],
    )
    monkeypatch.setattr(
        common,
        "gpu_compute_processes",
        lambda: [{
            "gpu_uuid": "GPU-4",
            "pid": 123,
            "process_name": "foreign-workload",
            "used_memory_mib": 512,
            "owner": "other-user",
        }],
    )

    report = common.preflight_gpus([4])

    assert report["compute_processes"][0]["owner"] == "other-user"
    assert report["require_idle"] is False


def test_controller_waits_for_busy_selected_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object.__new__(Controller)
    controller.config = {"health": {"require_idle_gpus": True}}
    controller.state = {}
    controller._save_state = lambda: None
    calls = 0
    events: list[str] = []

    def fake_preflight(gpu_ids, *, require_idle=False):
        nonlocal calls
        calls += 1
        assert list(gpu_ids) == [2, 3]
        assert require_idle is True
        if calls == 1:
            raise common.GpuBusyError(
                "selected GPUs are not idle: GPU 3 pid=123 owner=other-user"
            )
        return {"selected": [{"index": 2}, {"index": 3}]}

    monkeypatch.setattr("controller.preflight_gpus", fake_preflight)
    monkeypatch.setattr("controller.stop_requested", lambda: False)
    monkeypatch.setattr("controller.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "controller.append_event",
        lambda event, **_fields: events.append(event),
    )

    report = controller._preflight_stage_gpus([2, 3], "ARENA")

    assert calls == 2
    assert report == {"selected": [{"index": 2}, {"index": 3}]}
    assert events == [
        "gpu_preflight_waiting",
        "gpu_preflight_resumed",
        "gpu_preflight_passed",
    ]
    assert "gpu_wait_reason" not in controller.state
    assert "gpu_waiting_since" not in controller.state


def test_worker_error_summary_prefers_cuda_oom_over_trailing_hint() -> None:
    error = """Traceback (most recent call last):
torch.AcceleratorError: CUDA error: out of memory
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
"""

    assert _summarize_worker_error(error) == (
        "torch.AcceleratorError: CUDA error: out of memory"
    )


def test_dynamic_pool_refills_a_free_gpu_while_another_game_is_running() -> None:
    class FakeProcess:
        next_pid = 1000

        def __init__(self, polls_until_done: int) -> None:
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.polls_until_done = polls_until_done

        def poll(self) -> int | None:
            self.polls_until_done -= 1
            return 0 if self.polls_until_done <= 0 else None

    durations = {"slow": 5, "fast-1": 1, "fast-2": 1}
    launches: list[tuple[str, int]] = []
    completions: list[str] = []

    def launch(task: str, gpu_id: int) -> FakeProcess:
        launches.append((task, gpu_id))
        return FakeProcess(durations[task])

    outcomes, pending = run_dynamic_process_pool(
        tasks=["slow", "fast-1", "fast-2"],
        gpu_slots=[3, 4],
        launch=launch,
        on_complete=lambda outcome: completions.append(outcome.task),
        should_stop=lambda: False,
        on_snapshot=lambda _pids: None,
        sleep=lambda _seconds: None,
    )

    assert launches == [("slow", 3), ("fast-1", 4), ("fast-2", 4)]
    assert completions[:2] == ["fast-1", "fast-2"]
    assert [outcome.returncode for outcome in outcomes] == [0, 0, 0]
    assert pending == 0


def test_dynamic_pool_drains_running_games_without_launching_after_stop() -> None:
    class FakeProcess:
        next_pid = 2000

        def __init__(self, polls_until_done: int) -> None:
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.polls_until_done = polls_until_done

        def poll(self) -> int | None:
            self.polls_until_done -= 1
            return 0 if self.polls_until_done <= 0 else None

    durations = {"slow": 4, "fast": 1, "must-not-start": 1}
    launches: list[str] = []
    stop = False

    def launch(task: str, _gpu_id: int) -> FakeProcess:
        launches.append(task)
        return FakeProcess(durations[task])

    def complete(outcome) -> None:
        nonlocal stop
        if outcome.task == "fast":
            stop = True

    outcomes, pending = run_dynamic_process_pool(
        tasks=["slow", "fast", "must-not-start"],
        gpu_slots=[3, 4],
        launch=launch,
        on_complete=complete,
        should_stop=lambda: stop,
        on_snapshot=lambda _pids: None,
        sleep=lambda _seconds: None,
    )

    assert launches == ["slow", "fast"]
    assert {outcome.task for outcome in outcomes} == {"slow", "fast"}
    assert pending == 1


def test_dynamic_pool_allows_a_persistent_worker_pid_to_be_reused() -> None:
    class ReusedProcess:
        pid = 3000

        def __init__(self) -> None:
            self.ready = False

        def assign(self) -> None:
            self.ready = True

        def poll(self) -> int | None:
            if not self.ready:
                return None
            self.ready = False
            return 0

    worker = ReusedProcess()
    launches: list[tuple[str, int]] = []

    def launch(task: str, gpu_id: int) -> ReusedProcess:
        launches.append((task, gpu_id))
        worker.assign()
        return worker

    outcomes, pending = run_dynamic_process_pool(
        tasks=["game-1", "game-2"],
        gpu_slots=[2],
        launch=launch,
        on_complete=lambda _outcome: None,
        should_stop=lambda: False,
        on_snapshot=lambda _pids: None,
        sleep=lambda _seconds: None,
    )

    assert launches == [("game-1", 2), ("game-2", 2)]
    assert [outcome.process.pid for outcome in outcomes] == [3000, 3000]
    assert pending == 0


def test_dynamic_pool_cancels_active_games_after_gate_is_locked() -> None:
    class FakeProcess:
        next_pid = 5000

        def __init__(self, polls_until_done: int) -> None:
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.polls_until_done = polls_until_done

        def poll(self) -> int | None:
            self.polls_until_done -= 1
            return 0 if self.polls_until_done <= 0 else None

    locked = False
    canceled: list[str] = []

    def complete(outcome) -> None:
        nonlocal locked
        if outcome.task == "decider":
            locked = True

    outcomes, skipped = run_dynamic_process_pool(
        tasks=["slow-active", "decider", "never-started"],
        gpu_slots=[2, 3],
        launch=lambda task, _gpu: FakeProcess(
            1 if task == "decider" else 10
        ),
        on_complete=complete,
        should_stop=lambda: False,
        should_cancel_active=lambda: locked,
        cancel_active=lambda active: canceled.extend(
            item.task for item in active
        ),
        on_snapshot=lambda _pids: None,
        sleep=lambda _seconds: None,
    )

    assert [outcome.task for outcome in outcomes] == ["decider"]
    assert canceled == ["slow-active"]
    assert skipped == 2


def test_arena_tasks_are_dispatched_in_balanced_color_order() -> None:
    black_job = {"name": "black", "side": 0, "offset": 0}
    white_job = {"name": "white", "side": 1, "offset": 0}
    tasks = [
        *((black_job, index, 1) for index in range(4)),
        *((white_job, index, 1) for index in range(4)),
    ]

    ordered = _interleave_arena_game_tasks(tasks)

    assert [task[0]["side"] for task in ordered] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert sorted((task[0]["side"], task[1]) for task in ordered) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_katago_gatekeeper_irreversible_boundaries() -> None:
    assert gatekeeper_decision(99.5, 199, 200, 0.5) is None
    assert gatekeeper_decision(100.0, 199, 200, 0.5) == "accept"
    assert gatekeeper_decision(99.0, 200, 200, 0.5) == "reject"
    assert gatekeeper_decision(0.0, 101, 200, 0.5) == "reject"


def test_collection_and_arena_jobs_request_one_game_worker_per_process() -> None:
    controller = object.__new__(Controller)
    controller.config = load_config()
    champion = "/tmp/champion.pt"
    controller.state = {
        "champion_checkpoint": champion,
        "champion_iteration": 0,
        "champion_history": [{"iteration": 0, "checkpoint": champion}],
    }

    collection_jobs = controller._collection_jobs(iteration=1)
    arena_jobs = controller._arena_jobs(
        iteration=1,
        candidate="/tmp/candidate.pt",
        champion=champion,
    )

    assert collection_jobs
    assert arena_jobs
    assert {job["workers"] for job in collection_jobs} == {1}
    assert {job["workers"] for job in arena_jobs} == {1}
    assert len(collection_jobs) == 12
    assert sum(job["games"] for job in collection_jobs) == 150


def test_main_arena_uses_unpaired_seeds_when_configured() -> None:
    controller = object.__new__(Controller)
    controller.config = load_config()

    jobs = controller._arena_jobs(
        iteration=2,
        candidate="/tmp/candidate.pt",
        champion="/tmp/champion.pt",
    )

    black_seeds = {job["seed"] for job in jobs if job["side"] == 0}
    white_seeds = {job["seed"] for job in jobs if job["side"] == 1}
    assert black_seeds.isdisjoint(white_seeds)


@pytest.mark.parametrize(
    ("arena_update", "health_update", "message"),
    [
        ({"action_temperature": 0.0}, {}, "action_temperature"),
        ({"paired_seeds": True}, {}, "paired_seeds"),
        ({"persistent_workers": False}, {}, "early termination"),
        ({}, {"arena_min_unique_trajectory_rate": 0.49}, "unique trajectory"),
        ({}, {"arena_min_d4_opening_unique_rate": -0.01}, "D4 opening"),
    ],
)
def test_arena_config_fails_closed_on_duplicate_risk(
    arena_update: dict[str, object],
    health_update: dict[str, object],
    message: str,
) -> None:
    config = load_config()
    config["arena"] = {**config["arena"], **arena_update}
    config["health"] = {**config["health"], **health_update}

    with pytest.raises(ValueError, match=message):
        validate_arena_config(config)


def test_training_config_rejects_non_divisible_global_batch() -> None:
    config = load_config()
    config["training"] = {**config["training"], "batch_size": 769}

    with pytest.raises(ValueError, match="divisible"):
        validate_training_config(config)


def test_training_config_rejects_step_cap_below_minimum() -> None:
    config = load_config()
    config["training"] = {
        **config["training"],
        "max_steps_per_iteration": config["training"]["minimum_steps"] - 1,
    }

    with pytest.raises(ValueError, match="max_steps_per_iteration"):
        validate_training_config(config)


def test_gpu_stage_config_rejects_non_positive_process_chunks() -> None:
    config = load_config()
    config["collection"] = {**config["collection"], "games_per_process": 0}
    with pytest.raises(ValueError, match="games_per_process"):
        validate_gpu_stage_config(config)


def test_persistent_workers_require_single_game_dispatch() -> None:
    config = load_config()
    config["arena"] = {
        **config["arena"],
        "persistent_workers": True,
        "games_per_process": 2,
    }

    with pytest.raises(ValueError, match="single-game dispatch"):
        validate_gpu_stage_config(config)


def test_controller_launches_four_training_ranks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = object.__new__(Controller)
    controller.config = load_config()
    captured: dict[str, object] = {}

    def fake_run_logged(
        command: list[str],
        _log_path: Path,
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        captured["command"] = command
        captured["env"] = env
        return 1

    controller._run_logged = fake_run_logged  # type: ignore[method-assign]
    monkeypatch.setattr("controller.preflight_gpus", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("controller.append_event", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="training failed with code 1"):
        controller._train_candidate(9998, tmp_path / "manifest.txt", "/tmp/champion.pt")

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1:5] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
    ]
    assert "--resume" not in command
    assert "--stop-file" not in command
    assert command[command.index("--max-steps") + 1] == "6000"
    assert captured["env"] == {
        "CUDA_VISIBLE_DEVICES": "2,3,4,5",
        "OMP_NUM_THREADS": "4",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }


def test_collector_command_passes_the_job_worker_count() -> None:
    command = build_run_games_command(
        {
            "mode": "collect",
            "black": "/tmp/black.pt",
            "white": "/tmp/white.pt",
            "games": 8,
            "workers": 1,
            "seed": 1000,
            "offset": 16,
            "staging": Path("/tmp/staging"),
        },
        completed_games=2,
        save_name="experiments/example/staging",
        games_to_run=1,
    )

    assert command[command.index("--num-games") + 1] == "1"
    assert command[command.index("--num-workers") + 1] == "1"
    assert command[command.index("--seed") + 1] == "1002"
    assert command[command.index("--game-index-offset") + 1] == "18"


def test_chunked_game_invocations_preserve_legacy_per_game_seeds() -> None:
    invocations = _game_invocations(seed=1002, game_index_offset=18, num_games=3)
    assert invocations == [(1002, 18), (1003, 19), (1004, 20)]
    assert [seed + offset for seed, offset in invocations] == [1020, 1022, 1024]


def test_game_invocations_wrap_seeds_into_numpy_uint32_range() -> None:
    invocations = _game_invocations(
        seed=4_300_000_000,
        game_index_offset=0,
        num_games=3,
    )

    assert invocations == [(5_032_704, 0), (5_032_705, 1), (5_032_706, 2)]
    for seed, _offset in invocations:
        np.random.seed(seed)


def test_reused_search_agent_state_is_reset_between_games() -> None:
    class FakeAgent:
        _consec_below = 4
        _turns_played = 23
        last_search_result = object()

    agent = FakeAgent()
    _reset_reused_search_state(agent)

    assert agent._consec_below == 0
    assert agent._turns_played == 0
    assert agent.last_search_result is None


def test_persistent_worker_protocol_runs_one_task_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response = tmp_path / "response.json"
    game_log = tmp_path / "game.log"
    task = {
        "command": "run",
        "task_id": "arena-it0001-game0000003",
        "mode": "arena",
        "black_checkpoint": "/tmp/black.pt",
        "white_checkpoint": "/tmp/white.pt",
        "save_name": "experiments/example/arena",
        "seed": 1003,
        "game_index_offset": 3,
        "num_workers": 1,
        "log_path": str(game_log),
        "response_path": str(response),
    }
    seen: list[dict[str, object]] = []

    class FakeRunner:
        def run_task(self, payload: dict[str, object]) -> dict[str, object]:
            seen.append(payload)
            return {"winner": 2, "game_path": "/tmp/game0000003.npz"}

        def close(self) -> None:
            seen.append({"closed": True})

    monkeypatch.setattr("run_games.load_config", lambda: {})
    monkeypatch.setattr(
        "run_games.PersistentGameRunner",
        lambda _config, _mode: FakeRunner(),
    )
    stream = io.StringIO(
        json.dumps(task) + "\n" + json.dumps({"command": "shutdown"}) + "\n"
    )

    assert persistent_worker_loop("arena", input_stream=stream) == 0
    assert seen == [task, {"closed": True}]
    assert json.loads(response.read_text()) == {
        "error": None,
        "game_path": "/tmp/game0000003.npz",
        "ok": True,
        "pid": os.getpid(),
        "task_id": "arena-it0001-game0000003",
        "winner": 2,
    }


def test_persistent_runner_atomically_publishes_a_completed_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(PersistentGameRunner)
    runner.config = {}
    runner.mode = "arena"
    runner.agent_name_cache = {}
    task = {
        "command": "run",
        "task_id": "arena-it0001-game0000003",
        "mode": "arena",
        "black_checkpoint": "/tmp/black.pt",
        "white_checkpoint": "/tmp/white.pt",
        "save_name": "staging/candidate-black/shard000",
        "seed": 1003,
        "game_index_offset": 3,
        "num_workers": 1,
        "log_path": str(tmp_path / "game.log"),
    }
    worker_save_names: list[str] = []

    def fake_run(args, _config, *, agent_name_cache) -> None:
        assert agent_name_cache == {}
        worker_save_names.append(str(args.save_name))
        output = Path(args.save_name)
        if not output.is_absolute():
            output = tmp_path / output
        output.mkdir(parents=True, exist_ok=True)
        _write_arena_game(
            output / "20260806-000000-game0000003.npz",
            1,
            n_moves=2,
            trajectory_marker=3,
        )

    monkeypatch.setattr("run_games.GAME_DATA_ROOT", tmp_path)
    monkeypatch.setattr("run_games._run_bounded_games", fake_run)

    result = runner.run_task(task)

    published = tmp_path / str(task["save_name"]) / (
        "20260806-000000-game0000003.npz"
    )
    assert result == {"game_path": str(published), "winner": 1}
    assert "/.inflight/" in f"/{worker_save_names[0]}"
    assert published.is_file()
    assert not (tmp_path / str(task["save_name"]) / ".inflight").exists()


def test_partial_collection_resume_finds_index_gaps(tmp_path: Path) -> None:
    _write_mcts_npz(tmp_path / "20260726-000000-game0000016.npz")
    _write_mcts_npz(tmp_path / "20260726-000001-game0000018.npz")
    controller = object.__new__(Controller)
    controller.config = load_config()
    completed, _positions = controller._validate_existing_partial_progress(
        {
            "name": "gap-test",
            "games": 4,
            "offset": 16,
            "staging": tmp_path,
            "require_mcts": True,
            "allowed_visits": (100, 600),
        }
    )
    assert completed == {0, 2}


def test_partial_collection_progress_counts_staged_positions(tmp_path: Path) -> None:
    _write_mcts_npz(tmp_path / "20260726-000000-game0000016.npz")
    _write_mcts_npz(tmp_path / "20260726-000001-game0000018.npz")
    controller = object.__new__(Controller)
    controller.config = load_config()
    completed, positions = controller._validate_existing_partial_progress(
        {
            "name": "progress-test",
            "games": 4,
            "offset": 16,
            "staging": tmp_path,
            "require_mcts": True,
            "allowed_visits": (100, 600),
        }
    )
    assert completed == {0, 2}
    assert positions == 2


def test_atomic_manifest_and_checkpoint_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    atomic_write_text(manifest, "experiments/example/shard0\n")
    assert len(sha256_file(manifest)) == 64
    checkpoint = tmp_path / "resume.pt"
    atomic_torch_save({"model_state_dict": {"w": torch.ones(1)}, "iteration": 3}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert loaded["iteration"] == 3
    assert not checkpoint.with_name(checkpoint.name + ".tmp").exists()


def test_graceful_stop_flag(tmp_path: Path) -> None:
    flag = tmp_path / "stop.requested"
    assert not stop_requested(flag)
    flag.touch()
    assert stop_requested(flag)


def test_bootstrap_filename_index_and_duplicate_detection(tmp_path: Path) -> None:
    first = tmp_path / "20260725-000000-game0000007.npz"
    second = tmp_path / "20260725-000001-game0000007.npz"
    first.touch()
    assert extract_game_index(first) == 7
    second.touch()
    with pytest.raises(ValueError, match="duplicate bootstrap game index 7"):
        discover_game_indices(tmp_path)


def test_process_parallel_bootstrap_smoke(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["GAME_DATA_DIR"] = str(tmp_path)
    script = Path(__file__).with_name("bootstrap.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-games", "4",
            "--num-workers", "4",
            "--save-name", "smoke",
            "--seed", "0",
            "--game-index-offset", "0",
            "--max-moves", "2",
        ],
        check=True,
        cwd=script.parents[2],
        env=env,
        timeout=60,
    )
    files = discover_game_indices(tmp_path / "smoke")
    assert set(files) == {0, 1, 2, 3}
    for path in files.values():
        stats = validate_npz(path, board_size=9, require_mcts=False)
        assert stats["positions"] == 2


def _write_arena_game(
    path: Path,
    winner: int,
    n_moves: int = 200,
    *,
    termination: str = "double_pass",
    move_marker: int | None = None,
    trajectory_marker: int | None = None,
) -> None:
    moves = np.zeros((n_moves, 2), dtype=np.int16)
    if move_marker is not None:
        moves[0] = divmod(move_marker, 9)
    if trajectory_marker is not None:
        moves[-2] = divmod(trajectory_marker // 81, 9)
        moves[-1] = divmod(trajectory_marker % 81, 9)
    np.savez_compressed(
        path,
        boards=np.zeros((n_moves, 9, 9), dtype=np.int8),
        winner=np.int64(winner),
        num_moves=np.int64(n_moves),
        moves=moves,
        board_size=np.int64(9),
        termination=np.array(termination),
    )


def test_katago_gatekeeper_promotes_an_exact_50_percent_tie(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(100):
        _write_arena_game(
            black / f"g{index}.npz",
            1 if index < 50 else 2,
            trajectory_marker=index,
        )
        _write_arena_game(
            white / f"g{index}.npz",
            2 if index < 50 else 1,
            trajectory_marker=index + 100,
        )
    result = evaluate_arena(black, white)
    assert result["wins"] == 100
    assert result["score"] == pytest.approx(0.5)
    assert result["gatekeeper_decision"] == "accept"
    assert result["strength_gate"] is True
    assert result["promoted"] is True


def test_arena_stops_after_candidate_result_is_irreversible(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(50):
        _write_arena_game(
            black / f"g{index}.npz",
            1,
            trajectory_marker=index,
        )
        _write_arena_game(
            white / f"g{index}.npz",
            2,
            trajectory_marker=index + 50,
        )

    result = evaluate_arena(black, white)

    assert result["games"] == 100
    assert result["candidate_points"] == 100.0
    assert result["gatekeeper_decision"] == "accept"
    assert result["early_terminated"] is True
    assert result["remaining_games_skipped"] == 100
    assert result["promoted"] is True


def test_arena_does_not_stop_while_health_evidence_can_still_reverse(
    tmp_path: Path,
) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(50):
        _write_arena_game(black / f"g{index}.npz", 1)
        _write_arena_game(white / f"g{index}.npz", 2)

    with pytest.raises(ValueError, match="health evidence is still reversible"):
        evaluate_arena(black, white)


def test_arena_stops_after_baseline_result_is_irreversible(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(51):
        _write_arena_game(
            black / f"g{index}.npz",
            2,
            trajectory_marker=index,
        )
    for index in range(50):
        _write_arena_game(
            white / f"g{index}.npz",
            1,
            trajectory_marker=index + 51,
        )

    result = evaluate_arena(black, white)

    assert result["games"] == 101
    assert result["candidate_points"] == 0.0
    assert result["gatekeeper_decision"] == "reject"
    assert result["early_terminated"] is True
    assert result["remaining_games_skipped"] == 99
    assert result["promoted"] is False


def test_arena_rejects_an_incomplete_reversible_result(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(50):
        _write_arena_game(black / f"g{index}.npz", 1)
    for index in range(49):
        _write_arena_game(white / f"g{index}.npz", 2)

    with pytest.raises(ValueError, match="still reversible"):
        evaluate_arena(black, white)


def test_gatekeeper_publishes_and_resumes_an_irreversible_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = object.__new__(Controller)
    controller.config = load_config()
    controller.state = {"iteration": 7}
    controller._save_state = lambda: None
    jobs = controller._arena_jobs(
        iteration=7,
        candidate="/tmp/candidate.pt",
        champion="/tmp/champion.pt",
    )
    published_root = tmp_path / "published" / "arena-it0007"
    for job in jobs:
        label = "candidate-black" if job["side"] == 0 else "candidate-white"
        shard = str(job["name"]).rsplit("-", 1)[-1]
        job["staging"] = tmp_path / "staging" / label / shard
        job["final"] = published_root / label / shard
        job["log"] = tmp_path / "logs" / f"{job['name']}.log"

    calls: list[int] = []

    def fake_run(
        game_tasks,
        _gpu_slots,
        _omp_threads,
        arena_gate=None,
    ):
        calls.append(len(game_tasks))
        assert arena_gate is not None
        for job, game_index, _game_count in game_tasks[:100]:
            global_index = int(job["offset"]) + game_index
            path = Path(job["staging"]) / (
                f"20260806-000000-game{global_index:07d}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_arena_game(
                path,
                1 if int(job["side"]) == 0 else 2,
                n_moves=2,
                trajectory_marker=int(job["side"]) * 100 + global_index,
            )
            arena_gate.record_file(job, path)
        return [], max(0, len(game_tasks) - 100)

    monkeypatch.setattr(controller, "_run_persistent_game_tasks", fake_run)
    monkeypatch.setattr("controller.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("controller.append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("controller.preflight_gpus", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("controller.stop_requested", lambda: False)

    paths = controller._run_arena_gatekeeper_jobs(jobs, [2, 3, 4, 5], "ARENA")

    marker = json.loads(
        (tmp_path / "runtime" / "arena-it0007-early-stop.json").read_text()
    )
    assert marker["decision"] == "accept"
    assert marker["games"] == 100
    assert marker["remaining_games_skipped"] == 100
    assert controller.state["expected_games"] == 100
    assert controller.state["completed_games"] == 100
    assert len(paths) == 6
    result = evaluate_arena(
        published_root / "candidate-black",
        published_root / "candidate-white",
    )
    assert result["promoted"] is True
    assert result["early_terminated"] is True

    resumed_paths = controller._run_arena_gatekeeper_jobs(
        jobs,
        [2, 3, 4, 5],
        "ARENA",
    )
    assert resumed_paths == paths
    assert calls == [200, 0]


def test_arena_duplicate_trajectories_block_promotion(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(100):
        _write_arena_game(black / f"g{index}.npz", 1)
        _write_arena_game(white / f"g{index}.npz", 2)

    result = evaluate_arena(black, white)

    assert result["score"] == 1.0
    assert result["unique_trajectory_rate"] == pytest.approx(0.01)
    assert result["diversity_gate"] is False
    assert result["promoted"] is False


def test_arena_d4_opening_collapse_is_monitored_when_gate_disabled(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(100):
        _write_arena_game(
            black / f"g{index}.npz",
            1,
            trajectory_marker=index,
        )
        _write_arena_game(
            white / f"g{index}.npz",
            2,
            trajectory_marker=index + 100,
        )

    result = evaluate_arena(black, white)

    assert result["score"] == 1.0
    assert result["unique_trajectory_rate"] == 1.0
    assert result["diversity_gate"] is True
    assert result["candidate_as_black"]["d4_opening_unique"] == 1
    assert result["candidate_as_white"]["d4_opening_unique"] == 1
    assert result["d4_opening_diversity_gate"] is True
    assert result["promoted"] is True


def test_arena_max_moves_rate_does_not_block_promotion(tmp_path: Path) -> None:
    black = tmp_path / "candidate-black"
    white = tmp_path / "candidate-white"
    black.mkdir()
    white.mkdir()
    for index in range(100):
        _write_arena_game(
            black / f"g{index}.npz",
            1,
            termination="max_moves",
            move_marker=index,
        )
        _write_arena_game(
            white / f"g{index}.npz",
            2,
            termination="max_moves",
            move_marker=index + 100,
        )

    result = evaluate_arena(black, white)

    assert result["score"] == 1.0
    assert result["max_moves_rate"] == 1.0
    assert result["max_moves_health_gate"] is False
    assert result["promoted"] is True
