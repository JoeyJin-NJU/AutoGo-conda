#!/usr/bin/env python
"""Disk-only progress analysis; never talks to live workers."""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import EVENTS_PATH, EXP_DATA_ROOT, EXP_DIR, STATE_PATH, atomic_write_text


def summarize() -> dict:
    games = positions = black_wins = white_wins = draws = 0
    terminations: dict[str, int] = {}
    visit_budgets: dict[str, int] = {}
    teacher_positions = 0
    for path in sorted(EXP_DATA_ROOT.rglob("*.npz")) if EXP_DATA_ROOT.exists() else []:
        with np.load(path, allow_pickle=False) as data:
            games += 1
            positions += int(data["num_moves"])
            winner = int(data["winner"])
            black_wins += winner == 1
            white_wins += winner == 2
            draws += winner == 0
            termination = str(data["termination"])
            terminations[termination] = terminations.get(termination, 0) + 1
            if "mcts_visits" in data:
                totals = data["mcts_visits"].sum(axis=1)
                unique, counts = np.unique(totals, return_counts=True)
                for budget, count in zip(unique, counts):
                    key = str(int(budget))
                    visit_budgets[key] = visit_budgets.get(key, 0) + int(count)
                teacher_positions += int(data["is_teacher"].sum())
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    events = EVENTS_PATH.read_text().splitlines() if EVENTS_PATH.exists() else []
    promotions = [json.loads(line) for line in events if '"candidate_promoted"' in line]
    return {
        "games": games,
        "positions": positions,
        "black_wins": int(black_wins),
        "white_wins": int(white_wins),
        "draws": int(draws),
        "terminations": terminations,
        "visit_budgets": visit_budgets,
        "teacher_positions": teacher_positions,
        "current_stage": state.get("stage"),
        "current_iteration": state.get("iteration"),
        "champion_iteration": state.get("champion_iteration"),
        "promotions": promotions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write_report:
        report = [
            "# 9x9 komi-7 PCR training report",
            "",
            "## Current results",
            "",
            f"- Stage: {result['current_stage']}",
            f"- Iteration: {result['current_iteration']}",
            f"- Champion iteration: {result['champion_iteration']}",
            f"- Games: {result['games']}",
            f"- Positions: {result['positions']}",
            f"- Visit budgets: `{json.dumps(result['visit_budgets'], sort_keys=True)}`",
            f"- Teacher positions: {result['teacher_positions']}",
            f"- Terminations: `{json.dumps(result['terminations'], sort_keys=True)}`",
            f"- Promotions after bootstrap: {len(result['promotions'])}",
            "",
        ]
        atomic_write_text(EXP_DIR / "report.md", "\n".join(report))


if __name__ == "__main__":
    main()
