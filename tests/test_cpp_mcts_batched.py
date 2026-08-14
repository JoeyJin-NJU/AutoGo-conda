"""Parity check: run_simulations_batched (leaf-parallel + virtual loss) should
produce a root visit distribution close to the sequential run_simulations.

Virtual loss introduces some exploration noise, so we compare total-variation
distance rather than requiring exact match.
"""
from __future__ import annotations

import numpy as np
import pytest

import alpha_go_cpp


def biased_evaluator(state):
    """Deterministic policy evaluator: prefers lower flat indices."""
    moves = state.get_legal_moves_flat()
    moves.append(alpha_go_cpp.PASS_ACTION)
    weights = {m: 1.0 / (m + 2) for m in moves}
    total = sum(weights.values())
    return ({m: w / total for m, w in weights.items()}, 0.6)


def batched_from(single):
    """Wrap a single-board evaluator into a batched one."""
    return lambda states: [single(s) for s in states]


def _visit_distribution(tree) -> dict[int, float]:
    visits = tree.get_child_visit_counts()
    total = sum(visits.values())
    assert total > 0
    return {a: n / total for a, n in visits.items()}


def _tv_distance(p: dict[int, float], q: dict[int, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@pytest.mark.parametrize("leaf_batch_size", [4, 16, 64])
def test_batched_matches_sequential(leaf_batch_size: int) -> None:
    """Visit distribution from leaf-parallel MCTS should be close to sequential."""
    num_sims = 512

    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0  # disable noise for deterministic comparison

    # Sequential baseline
    board_a = alpha_go_cpp.GoBoard(9)
    tree_a = alpha_go_cpp.MCTSTree(board_a, cfg)
    tree_a.run_simulations(num_sims, biased_evaluator)
    dist_seq = _visit_distribution(tree_a)

    # Leaf-parallel
    board_b = alpha_go_cpp.GoBoard(9)
    tree_b = alpha_go_cpp.MCTSTree(board_b, cfg)
    tree_b.run_simulations_batched(num_sims, leaf_batch_size, batched_from(biased_evaluator))
    dist_par = _visit_distribution(tree_b)

    # Both should have the same total visit count at root.
    assert tree_a.get_root_visit_count() == num_sims
    assert tree_b.get_root_visit_count() == num_sims

    # TV distance should be small. Virtual loss spreads visits somewhat — the
    # larger the batch, the more aggressively siblings get explored — so the
    # tolerance scales with leaf_batch_size.
    tv = _tv_distance(dist_seq, dist_par)
    tol = 0.25 + 0.005 * leaf_batch_size
    assert tv < tol, f"TV(seq, batched[bs={leaf_batch_size}]) = {tv:.3f} >= tol {tol:.3f}"

    # Top-3 action set should overlap by at least one action — virtual loss
    # can shuffle the relative order of near-tied actions, but the sequential
    # top action should appear in the batched top-3 and vice versa.
    top3_seq = sorted(dist_seq, key=dist_seq.get, reverse=True)[:3]
    top3_par = sorted(dist_par, key=dist_par.get, reverse=True)[:3]
    assert set(top3_seq) & set(top3_par), (
        f"Top-3 disjoint: seq={top3_seq} vs batched={top3_par}"
    )


def test_batched_respects_pcr_sim_count() -> None:
    """When PCR is set, batched should still record exactly the sampled sim count as root visits."""
    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0
    cfg.pcr_sims = [32]
    cfg.pcr_probs = [1.0]

    board = alpha_go_cpp.GoBoard(9)
    tree = alpha_go_cpp.MCTSTree(board, cfg)
    # num_simulations argument is overridden by PCR; pass a different value to verify.
    tree.run_simulations_batched(999, 8, batched_from(biased_evaluator))
    assert tree.get_root_visit_count() == 32


def test_batched_clears_virtual_loss() -> None:
    """After run_simulations_batched returns, N_virt is implicit via PUCT scores —
    check that child visit counts sum to exactly num_sims (i.e. no ghost visits)."""
    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0

    board = alpha_go_cpp.GoBoard(9)
    tree = alpha_go_cpp.MCTSTree(board, cfg)
    tree.run_simulations_batched(128, 16, batched_from(biased_evaluator))

    total_child_visits = sum(tree.get_child_visit_counts().values())
    # Every root-descendant sim should increment exactly one child of root.
    assert total_child_visits == tree.get_root_visit_count() == 128
