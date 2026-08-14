#!/usr/bin/env python
"""Evaluate a color-balanced candidate/champion arena with health gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    D4_OPENING_PREFIX_DEPTHS,
    D4_OPENING_PREFIX_MOVES,
    atomic_write_json,
    canonical_d4_prefix,
    load_config,
)

_GATE_EPSILON = 1e-12


def gatekeeper_decision(
    candidate_points: float,
    games_played: int,
    max_games: int,
    required_score: float,
) -> str | None:
    """Return the irreversible KataGo-style strength decision, if any."""
    if max_games <= 0:
        raise ValueError("gatekeeper max_games must be positive")
    if games_played < 0 or games_played > max_games:
        raise ValueError("gatekeeper games_played must be within [0, max_games]")
    if not math.isfinite(candidate_points) or not 0.0 <= candidate_points <= games_played:
        raise ValueError("gatekeeper candidate_points is outside the played-game range")
    if not math.isfinite(required_score) or not 0.0 < required_score <= 1.0:
        raise ValueError("gatekeeper required_score must be within (0, 1]")

    required_points = max_games * required_score
    remaining = max_games - games_played
    if candidate_points + _GATE_EPSILON >= required_points:
        return "accept"
    if candidate_points + remaining + _GATE_EPSILON < required_points:
        return "reject"
    return None


def _trajectory_digest(moves: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(moves.shape).encode("ascii"))
    digest.update(str(moves.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(moves).tobytes())
    return digest.hexdigest()


def _read_side(root: Path, candidate_color: int, cfg: dict[str, Any]) -> dict[str, Any]:
    wins = losses = draws = positions = early_passes = early_positions = 0
    early_double_passes = max_moves = 0
    trajectory_digests: set[str] = set()
    d4_openings: dict[int, Counter[tuple[tuple[int, int], ...]]] = {
        depth: Counter() for depth in D4_OPENING_PREFIX_DEPTHS
    }
    files = sorted(root.rglob("*.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            winner = int(data["winner"])
            n_moves = int(data["num_moves"])
            moves = data["moves"]
            termination = str(data["termination"])
            trajectory_digests.add(_trajectory_digest(moves))
            for depth in D4_OPENING_PREFIX_DEPTHS:
                if n_moves >= depth and len(moves) >= depth:
                    d4_openings[depth][
                        canonical_d4_prefix(
                            np.asarray(moves[:depth]),
                            int(cfg["rules"]["board_size"]),
                        )
                    ] += 1
            if winner == candidate_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            positions += n_moves
            cutoff = min(n_moves, int(cfg["health"]["early_pass_before_ply"]))
            early_passes += int(np.all(moves[:cutoff] < 0, axis=1).sum())
            early_positions += cutoff
            if termination == "double_pass" and n_moves < int(
                cfg["health"]["early_double_pass_before_ply"]
            ):
                early_double_passes += 1
            if termination == "max_moves":
                max_moves += 1
    d4_opening_by_depth = {}
    for depth in D4_OPENING_PREFIX_DEPTHS:
        counts = d4_openings[depth]
        eligible = sum(counts.values())
        largest = max(counts.values(), default=0)
        d4_opening_by_depth[str(depth)] = {
            "eligible": eligible,
            "unique": len(counts),
            "unique_rate": len(counts) / max(1, eligible),
            "largest_cluster": largest,
            "largest_cluster_rate": largest / max(1, eligible),
        }
    d4_eligible = d4_opening_by_depth[str(D4_OPENING_PREFIX_MOVES)]["eligible"]
    d4_unique = d4_opening_by_depth[str(D4_OPENING_PREFIX_MOVES)]["unique"]
    d4_largest = d4_opening_by_depth[str(D4_OPENING_PREFIX_MOVES)][
        "largest_cluster"
    ]
    return {
        "games": len(files),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "positions": positions,
        "early_passes": early_passes,
        "early_positions": early_positions,
        "early_double_passes": early_double_passes,
        "max_moves": max_moves,
        "unique_trajectories": len(trajectory_digests),
        "unique_trajectory_rate": len(trajectory_digests) / max(1, len(files)),
        "d4_opening_eligible": d4_eligible,
        "d4_opening_unique": d4_unique,
        "d4_opening_unique_rate": d4_unique / max(1, d4_eligible),
        "d4_opening_largest_cluster": d4_largest,
        "d4_opening_largest_cluster_rate": d4_largest / max(1, d4_eligible),
        "d4_opening_by_depth": d4_opening_by_depth,
    }


def evaluate_arena(candidate_black_root: Path, candidate_white_root: Path) -> dict[str, Any]:
    cfg = load_config()
    black = _read_side(candidate_black_root, 1, cfg)
    white = _read_side(candidate_white_root, 2, cfg)
    total = black["games"] + white["games"]
    expected = int(cfg["arena"]["games"])
    expected_black = int(cfg["arena"]["candidate_as_black"])
    expected_white = int(cfg["arena"]["candidate_as_white"])
    if expected_black + expected_white != expected:
        raise ValueError("arena color quotas do not sum to the configured game count")
    if total <= 0:
        raise ValueError("arena has no completed games")
    if black["games"] > expected_black:
        raise ValueError(
            f"candidate-black arena games={black['games']}, maximum {expected_black}"
        )
    if white["games"] > expected_white:
        raise ValueError(
            f"candidate-white arena games={white['games']}, maximum {expected_white}"
        )
    if total > expected:
        raise ValueError(f"arena games={total}, maximum {expected}")
    wins = black["wins"] + white["wins"]
    losses = black["losses"] + white["losses"]
    draws = black["draws"] + white["draws"]
    candidate_points = wins + float(cfg["arena"]["draw_score"]) * draws
    threshold = float(cfg["arena"]["promotion_threshold"])
    strength_decision = gatekeeper_decision(
        candidate_points,
        total,
        expected,
        threshold,
    )
    early_terminated = total < expected
    if early_terminated and not bool(
        cfg["arena"].get("early_terminate_irreversible", False)
    ):
        raise ValueError(f"arena games={total}, expected {expected}")
    if strength_decision is None:
        raise ValueError(
            f"arena result is still reversible after {total}/{expected} games"
        )
    score = candidate_points / total
    early_passes = black["early_passes"] + white["early_passes"]
    early_positions = black["early_positions"] + white["early_positions"]
    early_pass_rate = early_passes / max(1, early_positions)
    early_double_passes = black["early_double_passes"] + white["early_double_passes"]
    early_double_pass_rate = early_double_passes / total
    max_moves = black["max_moves"] + white["max_moves"]
    max_moves_rate = max_moves / total
    unique_trajectories = (
        black["unique_trajectories"] + white["unique_trajectories"]
    )
    unique_trajectory_rate = unique_trajectories / total
    strength_gate = strength_decision == "accept"
    pass_gate = early_double_pass_rate <= float(
        cfg["health"]["early_double_pass_rate_blocks_promotion"]
    )
    max_moves_gate = max_moves_rate <= float(
        cfg["health"]["max_moves_rate_blocks_promotion"]
    )
    diversity_gate = unique_trajectory_rate >= float(
        cfg["health"]["arena_min_unique_trajectory_rate"]
    )
    minimum_d4_rate = float(cfg["health"]["arena_min_d4_opening_unique_rate"])
    d4_opening_diversity_gate = all(
        side["d4_opening_unique_rate"] >= minimum_d4_rate
        for side in (black, white)
    )
    health_evidence_irreversible = (
        not early_terminated
        or strength_decision == "reject"
        or (
            unique_trajectories + _GATE_EPSILON
            >= float(cfg["health"]["arena_min_unique_trajectory_rate"])
            * expected
            and minimum_d4_rate == 0.0
        )
    )
    if not health_evidence_irreversible:
        raise ValueError(
            "arena strength result is fixed but promotion health evidence "
            "is still reversible"
        )
    failed_gates = [
        name
        for name, passed in (
            ("strength_gate_failed", strength_gate),
            ("trajectory_diversity_gate_failed", diversity_gate),
            ("d4_opening_diversity_gate_failed", d4_opening_diversity_gate),
        )
        if not passed
    ]
    promoted = not failed_gates
    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "candidate_points": candidate_points,
        "score": score,
        "threshold": threshold,
        "candidate_wins_ties": True,
        "gatekeeper_decision": strength_decision,
        "early_terminated": early_terminated,
        "remaining_games_skipped": expected - total,
        "health_evidence_irreversible": health_evidence_irreversible,
        "candidate_as_black": black,
        "candidate_as_white": white,
        "early_pass_rate": early_pass_rate,
        "early_double_pass_rate": early_double_pass_rate,
        "max_moves_rate": max_moves_rate,
        "max_moves_rate_limit": float(
            cfg["health"]["max_moves_rate_blocks_promotion"]
        ),
        "unique_trajectories": unique_trajectories,
        "unique_trajectory_rate": unique_trajectory_rate,
        "minimum_unique_trajectory_rate": float(
            cfg["health"]["arena_min_unique_trajectory_rate"]
        ),
        "strength_gate": strength_gate,
        "pass_health_gate": pass_gate,
        "max_moves_health_gate": max_moves_gate,
        "diversity_gate": diversity_gate,
        "d4_opening_diversity_gate": d4_opening_diversity_gate,
        "minimum_d4_opening_unique_rate": minimum_d4_rate,
        "promoted": promoted,
        "decision_reason": (
            "strength_gate_passed"
            if promoted
            else ",".join(failed_gates)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-black-root", type=Path, required=True)
    parser.add_argument("--candidate-white-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_arena(args.candidate_black_root, args.candidate_white_root)
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
