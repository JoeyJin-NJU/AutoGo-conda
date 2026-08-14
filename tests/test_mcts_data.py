"""Tests for MCTS data storage and action probability parity.

Verifies that:
1. Sparse visit counts from C++ MCTSTree convert correctly to dense arrays
2. Python recomputation of action probabilities from stored N(s,a) and temperature
   matches MCTSTree::get_action_probabilities exactly
3. Round-trip save/load of MCTS data through NPZ files preserves correctness
4. GoDataset returns label-smoothed fallback for files without MCTS data
"""
from __future__ import annotations

from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch

from alpha_go.dataset import GoDataset
from alpha_go.gameplay import GameRecord, MoveMetric, save_game_data


BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1  # 82: 81 board positions + pass
PASS_IDX = N_ACTIONS - 1


def dummy_evaluator(board: alpha_go_cpp.GoBoard):
    """Uniform policy + value 0.5 evaluator for testing."""
    moves = board.get_legal_moves_flat()
    moves.append(alpha_go_cpp.PASS_ACTION)
    prob = 1.0 / len(moves)
    policy = {m: prob for m in moves}
    return (policy, 0.5)


def sparse_to_dense_visits(sparse_visits: dict[int, int], board_size: int) -> np.ndarray:
    """Convert sparse visit counts dict to dense int16 array (same as gameplay.py)."""
    n_actions = board_size * board_size + 1
    pass_idx = n_actions - 1
    dense = np.zeros(n_actions, dtype=np.int16)
    for flat_idx, count in sparse_visits.items():
        if flat_idx == alpha_go_cpp.PASS_ACTION:
            dense[pass_idx] = count
        else:
            dense[flat_idx] = count
    return dense


def recompute_action_probs(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
    """Recompute action probabilities from visit counts and temperature.

    Matches MCTSTree::get_action_probabilities in mcts.cpp.
    """
    if temperature == 0:
        probs = np.zeros_like(visit_counts, dtype=np.float32)
        if visit_counts.sum() > 0:
            probs[np.argmax(visit_counts)] = 1.0
        return probs

    visits_f = visit_counts.astype(np.float32)
    visits_temp = np.power(visits_f, 1.0 / temperature)
    total = visits_temp.sum()
    if total > 0:
        return visits_temp / total
    return np.zeros_like(visits_f)


def sparse_probs_to_dense(sparse_probs: dict[int, float], board_size: int) -> np.ndarray:
    """Convert sparse probability dict from C++ to dense array for comparison."""
    n_actions = board_size * board_size + 1
    pass_idx = n_actions - 1
    dense = np.zeros(n_actions, dtype=np.float32)
    for flat_idx, prob in sparse_probs.items():
        if flat_idx == alpha_go_cpp.PASS_ACTION:
            dense[pass_idx] = prob
        else:
            dense[flat_idx] = prob
    return dense


class TestActionProbabilityParity:
    """Verify Python recomputation matches C++ MCTSTree::get_action_probabilities."""

    @pytest.fixture
    def mcts_tree(self):
        """Create an MCTSTree with 64 simulations."""
        board = alpha_go_cpp.GoBoard(BOARD_SIZE)
        config = alpha_go_cpp.MCTSConfig()
        config.c_puct = 1.5
        config.dirichlet_alpha = 0.0
        config.temperature = 1.0
        tree = alpha_go_cpp.MCTSTree(board, config)
        tree.run_simulations(64, dummy_evaluator)
        return tree

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_action_probs_match(self, mcts_tree, temperature):
        """Recomputed action probs from dense visit counts match C++ tree."""
        # Get sparse visit counts from C++ and convert to dense
        sparse_visits = mcts_tree.get_child_visit_counts()
        dense_visits = sparse_to_dense_visits(sparse_visits, BOARD_SIZE)

        # Recompute in Python
        python_probs = recompute_action_probs(dense_visits, temperature)

        # Get C++ reference
        cpp_sparse_probs = mcts_tree.get_action_probabilities(temperature)
        cpp_dense_probs = sparse_probs_to_dense(cpp_sparse_probs, BOARD_SIZE)

        np.testing.assert_allclose(python_probs, cpp_dense_probs, rtol=1e-5, atol=1e-7)

    def test_temperature_zero_argmax(self, mcts_tree):
        """Temperature=0 picks the action with most visits in both implementations."""
        sparse_visits = mcts_tree.get_child_visit_counts()
        dense_visits = sparse_to_dense_visits(sparse_visits, BOARD_SIZE)

        python_probs = recompute_action_probs(dense_visits, temperature=0)
        cpp_sparse_probs = mcts_tree.get_action_probabilities(0)
        cpp_dense_probs = sparse_probs_to_dense(cpp_sparse_probs, BOARD_SIZE)

        # Both should have exactly one 1.0 at the same index
        assert python_probs.sum() == pytest.approx(1.0)
        assert np.argmax(python_probs) == np.argmax(cpp_dense_probs)

    def test_visit_counts_sum(self, mcts_tree):
        """Total visit counts should equal num_simulations (approximately)."""
        sparse_visits = mcts_tree.get_child_visit_counts()
        dense_visits = sparse_to_dense_visits(sparse_visits, BOARD_SIZE)
        # Each simulation visits root once, so children get ~64 total visits
        assert dense_visits.sum() > 0
        assert dense_visits.sum() <= 64  # can't exceed total sims

    def test_pass_action_mapping(self, mcts_tree):
        """PASS_ACTION (-1) in C++ maps to index board_size^2 in dense array."""
        sparse_visits = mcts_tree.get_child_visit_counts()
        dense_visits = sparse_to_dense_visits(sparse_visits, BOARD_SIZE)

        if alpha_go_cpp.PASS_ACTION in sparse_visits:
            assert dense_visits[PASS_IDX] == sparse_visits[alpha_go_cpp.PASS_ACTION]

        # Verify no negative indices made it into the array
        # All visits should be non-negative
        assert (dense_visits >= 0).all()

    def test_unvisited_actions_are_zero(self, mcts_tree):
        """Actions not visited in MCTS have 0 in the dense array."""
        sparse_visits = mcts_tree.get_child_visit_counts()
        dense_visits = sparse_to_dense_visits(sparse_visits, BOARD_SIZE)

        visited_indices = set()
        for flat_idx in sparse_visits:
            if flat_idx == alpha_go_cpp.PASS_ACTION:
                visited_indices.add(PASS_IDX)
            else:
                visited_indices.add(flat_idx)

        for i in range(N_ACTIONS):
            if i not in visited_indices:
                assert dense_visits[i] == 0

    def test_q_values_sparse_to_dense(self, mcts_tree):
        """Q values convert correctly from sparse to dense."""
        sparse_q = mcts_tree.get_child_q_values()
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        pass_idx = n_actions - 1

        dense_q = np.zeros(n_actions, dtype=np.float32)
        for flat_idx, q in sparse_q.items():
            if flat_idx == alpha_go_cpp.PASS_ACTION:
                dense_q[pass_idx] = q
            else:
                dense_q[flat_idx] = q

        # Q values should be in [0, 1] for visited actions
        for flat_idx, q in sparse_q.items():
            idx = pass_idx if flat_idx == alpha_go_cpp.PASS_ACTION else flat_idx
            assert dense_q[idx] == pytest.approx(q)
            assert 0.0 <= q <= 1.0


class TestMCTSDataRoundTrip:
    """Test saving and loading MCTS data through NPZ files."""

    def test_save_load_mcts_data(self, tmp_path):
        """MCTS visit counts and Q values survive NPZ round-trip."""
        # Create a GameRecord with mock MCTS metrics
        n_moves = 5
        metrics = []
        for i in range(n_moves):
            visits = np.zeros(N_ACTIONS, dtype=np.int16)
            visits[0] = 10  # top-left gets 10 visits
            visits[40] = 20  # center gets 20 visits
            visits[PASS_IDX] = 2  # pass gets 2 visits

            q_vals = np.zeros(N_ACTIONS, dtype=np.float32)
            q_vals[0] = 0.3
            q_vals[40] = 0.7
            q_vals[PASS_IDX] = 0.1

            metrics.append(MoveMetric(
                move_step=i,
                policy_entropy=None,
                agent_color="black" if i % 2 == 0 else "white",
                visit_counts=visits,
                q_values=q_vals,
                temperature=1.0,
                root_value=0.45,
            ))

        record = GameRecord(
            board_size=BOARD_SIZE,
            black_agent="test_black",
            white_agent="test_white",
            moves=[(0, 0)] * n_moves,
            boards=[np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)] * n_moves,
            move_metrics=metrics,
            winner=1,
            result="B+5.5",
            num_moves=n_moves,
            termination="double_pass",
        )

        # Save
        output_dir = tmp_path / "test_games"
        output_dir.mkdir()
        filepath = save_game_data(record, output_dir, game_index=0, date_slug="test")

        # Load and verify
        data = dict(np.load(filepath))
        assert "mcts_visits" in data
        assert "mcts_q_values" in data
        assert "mcts_temperatures" in data
        assert "mcts_root_values" in data

        assert data["mcts_visits"].shape == (n_moves, N_ACTIONS)
        assert data["mcts_q_values"].shape == (n_moves, N_ACTIONS)
        assert data["mcts_temperatures"].shape == (n_moves,)
        assert data["mcts_root_values"].shape == (n_moves,)

        # Check values survived
        assert data["mcts_visits"][0, 0] == 10
        assert data["mcts_visits"][0, 40] == 20
        assert data["mcts_visits"][0, PASS_IDX] == 2
        assert data["mcts_q_values"][0, 40] == pytest.approx(0.7)
        assert data["mcts_temperatures"][0] == pytest.approx(1.0)
        assert data["mcts_root_values"][0] == pytest.approx(0.45)

    def test_recompute_policy_from_saved(self, tmp_path):
        """Action probs recomputed from saved visits match expected values."""
        visits = np.zeros(N_ACTIONS, dtype=np.int16)
        visits[10] = 30
        visits[20] = 10
        visits[PASS_IDX] = 5
        temperature = 1.0

        metrics = [MoveMetric(
            move_step=0,
            policy_entropy=None,
            agent_color="black",
            visit_counts=visits,
            q_values=np.zeros(N_ACTIONS, dtype=np.float32),
            temperature=temperature,
            root_value=0.5,
        )]

        record = GameRecord(
            board_size=BOARD_SIZE,
            black_agent="test",
            white_agent="test",
            moves=[(1, 1)],
            boards=[np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)],
            move_metrics=metrics,
            winner=1,
            result="B+1.5",
            num_moves=1,
            termination="double_pass",
        )

        output_dir = tmp_path / "test_games"
        output_dir.mkdir()
        filepath = save_game_data(record, output_dir, game_index=0, date_slug="test")

        # Load and recompute
        data = np.load(filepath)
        loaded_visits = data["mcts_visits"][0].astype(np.float32)
        loaded_temp = float(data["mcts_temperatures"][0])

        recomputed = recompute_action_probs(loaded_visits, loaded_temp)

        # With temperature=1.0, probs should be proportional to visits
        # 30/(30+10+5) = 0.6667, 10/45 = 0.2222, 5/45 = 0.1111
        assert recomputed[10] == pytest.approx(30.0 / 45.0, rel=1e-5)
        assert recomputed[20] == pytest.approx(10.0 / 45.0, rel=1e-5)
        assert recomputed[PASS_IDX] == pytest.approx(5.0 / 45.0, rel=1e-5)
        assert recomputed.sum() == pytest.approx(1.0)

    def test_no_mcts_data_backward_compatible(self, tmp_path):
        """Games without MCTS data save fine (no mcts_ keys in NPZ)."""
        record = GameRecord(
            board_size=BOARD_SIZE,
            black_agent="test",
            white_agent="test",
            moves=[(0, 0)],
            boards=[np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)],
            winner=1,
            result="B+1.5",
            num_moves=1,
            termination="double_pass",
        )

        output_dir = tmp_path / "test_games"
        output_dir.mkdir()
        filepath = save_game_data(record, output_dir, game_index=0, date_slug="test")

        data = dict(np.load(filepath))
        assert "mcts_visits" not in data
        assert "mcts_q_values" not in data


def _make_game_npz(output_dir: Path, game_index: int, n_moves: int = 4,
                   with_mcts: bool = False, mcts_temperature: float = 1.0) -> Path:
    """Helper to create a game NPZ file for GoDataset tests."""
    moves = [(i % BOARD_SIZE, (i + 1) % BOARD_SIZE) for i in range(n_moves)]
    boards = [np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8) for _ in range(n_moves)]

    metrics = None
    if with_mcts:
        metrics = []
        for i in range(n_moves):
            visits = np.zeros(N_ACTIONS, dtype=np.int16)
            visits[10] = 30
            visits[20] = 10
            metrics.append(MoveMetric(
                move_step=i, policy_entropy=None,
                agent_color="black" if i % 2 == 0 else "white",
                visit_counts=visits,
                q_values=np.zeros(N_ACTIONS, dtype=np.float32),
                temperature=mcts_temperature, root_value=0.5,
            ))

    record = GameRecord(
        board_size=BOARD_SIZE, black_agent="test", white_agent="test",
        moves=moves, boards=boards,
        move_metrics=metrics or [],
        winner=1, result="B+1.5", num_moves=n_moves, termination="double_pass",
    )
    return save_game_data(record, output_dir, game_index=game_index, date_slug="test")


class TestGoDatasetMCTSPolicy:
    """Test GoDataset with load_mcts_policy flag."""

    def test_mcts_policy_from_visits(self, tmp_path):
        """GoDataset returns mcts_policy computed from visit counts."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        _make_game_npz(game_dir, 0, n_moves=4, with_mcts=True)

        ds = GoDataset(game_dir, load_mcts_policy=True)
        sample = ds[0]  # First position (black's move)

        assert "mcts_policy" in sample
        assert "has_mcts" in sample
        assert sample["has_mcts"] is True
        assert sample["mcts_policy"].shape == (N_ACTIONS,)
        assert sample["mcts_policy"].sum() == pytest.approx(1.0, abs=1e-5)
        # Visit counts were 30 at idx 10, 10 at idx 20 -> probs 0.75, 0.25
        assert sample["mcts_policy"][10] == pytest.approx(0.75, rel=1e-4)
        assert sample["mcts_policy"][20] == pytest.approx(0.25, rel=1e-4)

    def test_policy_target_temperature_is_independent_of_action_temperature(self, tmp_path):
        """A fixed target temperature preserves visits despite a cold played move."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        _make_game_npz(
            game_dir,
            0,
            n_moves=1,
            with_mcts=True,
            mcts_temperature=0.15,
        )

        legacy = GoDataset(game_dir, load_mcts_policy=True)[0]["mcts_policy"]
        decoupled = GoDataset(
            game_dir,
            load_mcts_policy=True,
            policy_target_temperature=1.0,
        )[0]["mcts_policy"]

        assert legacy[10] > 0.999
        assert decoupled[10] == pytest.approx(0.75, rel=1e-4)
        assert decoupled[20] == pytest.approx(0.25, rel=1e-4)

    def test_label_smoothed_fallback(self, tmp_path):
        """GoDataset returns label-smoothed one-hot when no MCTS data in NPZ."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        _make_game_npz(game_dir, 0, n_moves=4, with_mcts=False)

        ds = GoDataset(game_dir, load_mcts_policy=True)
        sample = ds[0]  # First position, move is (0, 1)

        assert "mcts_policy" in sample
        assert "has_mcts" in sample
        assert sample["has_mcts"] is False
        assert sample["mcts_policy"].shape == (N_ACTIONS,)
        assert sample["mcts_policy"].sum() == pytest.approx(1.0, abs=1e-5)

        # Ground truth move (0, 1) -> flat index 1
        target_idx = 0 * BOARD_SIZE + 1
        smooth_eps = 0.1
        expected_peak = 1.0 - smooth_eps + smooth_eps / N_ACTIONS
        assert sample["mcts_policy"][target_idx] == pytest.approx(expected_peak, rel=1e-4)
        # Other actions get smooth_eps / n_actions
        other_idx = 50  # arbitrary non-target
        assert sample["mcts_policy"][other_idx] == pytest.approx(smooth_eps / N_ACTIONS, rel=1e-4)

    def test_mixed_dataset(self, tmp_path):
        """GoDataset handles mix of files with and without MCTS data."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        _make_game_npz(game_dir, 0, n_moves=2, with_mcts=True)
        _make_game_npz(game_dir, 1, n_moves=2, with_mcts=False)

        ds = GoDataset(game_dir, load_mcts_policy=True)
        assert len(ds) == 4  # 2 moves per game, 2 games

        # First game has MCTS data
        s0 = ds[0]
        assert s0["has_mcts"] is True

        # Second game does not
        s2 = ds[2]
        assert s2["has_mcts"] is False
        assert s2["mcts_policy"].sum() == pytest.approx(1.0, abs=1e-5)

    def test_no_mcts_policy_flag(self, tmp_path):
        """GoDataset without load_mcts_policy returns no mcts fields."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        _make_game_npz(game_dir, 0, n_moves=2, with_mcts=True)

        ds = GoDataset(game_dir, load_mcts_policy=False)
        sample = ds[0]
        assert "mcts_policy" not in sample
        assert "has_mcts" not in sample
