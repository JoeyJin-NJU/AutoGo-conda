#!/usr/bin/env python
"""Read-only terminal dashboard for the live experiment."""
from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from arena import gatekeeper_decision
from common import (
    CHECKPOINT_ROOT,
    D4_OPENING_PREFIX_DEPTHS,
    D4_OPENING_PREFIX_MOVES,
    EVENTS_PATH,
    EXP_DATA_ROOT,
    EXP_NAME,
    LOGS_DIR,
    RUNTIME_DIR,
    STATE_PATH,
    STOP_PATH,
    canonical_d4_prefix,
    disk_free_gib,
    gpu_compute_processes,
    gpu_inventory,
    load_config,
)

ITER_RE = re.compile(r"it(\d{4})")
TRAIN_RESULT_MARKER = "===RESULT==="
def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists() or lines <= 0:
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    return subprocess.run(
        ["ps", "-p", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _timestamp_age(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return time.time() - parsed.timestamp()
    except ValueError:
        return None


def _iteration_from_name(path: Path) -> int:
    match = ITER_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _scan_npz_roots(roots: Iterable[Path], *, board_size: int = 9) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "positions": 0,
            "invalid": 0,
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
        }
    )
    d4_openings: dict[
        str,
        dict[int, Counter[tuple[tuple[int, int], ...]]],
    ] = defaultdict(lambda: defaultdict(Counter))
    games = positions = invalid = 0
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in sorted(root.rglob("*.npz"))
                if ".inflight" not in path.relative_to(root).parts
            )
    for path in files:
        root = next(root for root in roots if path.is_relative_to(root))
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else root.name
        try:
            with np.load(path, allow_pickle=False) as data:
                n_moves = int(data["num_moves"])
                winner = int(data["winner"])
                if winner not in (0, 1, 2):
                    raise ValueError(f"invalid winner {winner}")
                moves = np.asarray(data["moves"])
                for depth in D4_OPENING_PREFIX_DEPTHS:
                    if n_moves >= depth and len(moves) >= depth:
                        d4_openings[group][depth][
                            canonical_d4_prefix(moves[:depth], board_size)
                        ] += 1
        except (OSError, ValueError, KeyError):
            invalid += 1
            groups[group]["invalid"] += 1
            continue
        games += 1
        positions += n_moves
        groups[group]["games"] += 1
        groups[group]["positions"] += n_moves
        winner_key = {0: "draws", 1: "black_wins", 2: "white_wins"}[winner]
        groups[group][winner_key] += 1
    group_rows = []
    for name, stats in sorted(groups.items()):
        opening_by_depth = {}
        for depth in D4_OPENING_PREFIX_DEPTHS:
            counts = d4_openings[name][depth]
            depth_eligible = sum(counts.values())
            depth_largest = max(counts.values(), default=0)
            opening_by_depth[str(depth)] = {
                "eligible": depth_eligible,
                "unique": len(counts),
                "unique_rate": len(counts) / max(1, depth_eligible),
                "largest_cluster": depth_largest,
                "largest_cluster_rate": depth_largest / max(1, depth_eligible),
            }
        opening_counts = d4_openings[name][D4_OPENING_PREFIX_MOVES]
        eligible = sum(opening_counts.values())
        largest = max(opening_counts.values(), default=0)
        group_rows.append({
            "name": name,
            **stats,
            "d4_opening_eligible": eligible,
            "d4_opening_unique": len(opening_counts),
            "d4_opening_unique_rate": len(opening_counts) / max(1, eligible),
            "d4_opening_largest_cluster": largest,
            "d4_opening_largest_cluster_rate": largest / max(1, eligible),
            "d4_opening_by_depth": opening_by_depth,
        })
    return {
        "games": games,
        "positions": positions,
        "invalid": invalid,
        "groups": group_rows,
    }


def _arena_live_result(
    groups: Iterable[dict[str, Any]],
    *,
    draw_score: float,
    promotion_threshold: float,
    max_games: int = 200,
) -> dict[str, Any]:
    by_name = {str(group["name"]): group for group in groups}
    sides: dict[str, dict[str, Any]] = {}
    for name, candidate_winner, opponent_winner in (
        ("candidate-black", "black_wins", "white_wins"),
        ("candidate-white", "white_wins", "black_wins"),
    ):
        group = by_name.get(name, {})
        wins = int(group.get(candidate_winner, 0))
        losses = int(group.get(opponent_winner, 0))
        draws = int(group.get("draws", 0))
        games = wins + losses + draws
        sides[name] = {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": (wins + draw_score * draws) / games if games else None,
        }

    games = sum(int(side["games"]) for side in sides.values())
    wins = sum(int(side["wins"]) for side in sides.values())
    losses = sum(int(side["losses"]) for side in sides.values())
    draws = sum(int(side["draws"]) for side in sides.values())
    candidate_points = wins + draw_score * draws
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "candidate_points": candidate_points,
        "score": candidate_points / games if games else None,
        "promotion_threshold": promotion_threshold,
        "gatekeeper_decision": (
            gatekeeper_decision(
                candidate_points,
                games,
                max_games,
                promotion_threshold,
            )
            if games
            else None
        ),
        "candidate_as_black": sides["candidate-black"],
        "candidate_as_white": sides["candidate-white"],
    }


def _current_stage_scan(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    iteration = state.get("iteration")
    if iteration is None:
        return {"kind": None, "games": 0, "positions": 0, "invalid": 0, "groups": []}
    stage = str(state.get("stage", ""))
    expected = int(state.get("expected_games") or 0)
    arena_expected = int(config["arena"]["games"])
    is_arena = stage == "ARENA" or (
        stage in {"FAILED", "STOPPED"} and expected == arena_expected
    )
    kind = "arena" if is_arena else "collect"
    name = f"{kind}-it{int(iteration):04d}"
    scan = _scan_npz_roots(
        [EXP_DATA_ROOT / name, EXP_DATA_ROOT / ".staging" / name],
        board_size=int(config["rules"]["board_size"]),
    )
    scan["kind"] = kind
    scan["name"] = name
    if is_arena:
        scan["arena_live"] = _arena_live_result(
            scan["groups"],
            draw_score=float(config["arena"]["draw_score"]),
            promotion_threshold=float(config["arena"]["promotion_threshold"]),
            max_games=arena_expected,
        )
    return scan


def _arena_history() -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for path in sorted(
        RUNTIME_DIR.glob("arena-it[0-9][0-9][0-9][0-9].json"),
        key=_iteration_from_name,
    ):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        history.append({"iteration": _iteration_from_name(path), **result})
    return history


def _latest_training_result() -> dict[str, Any] | None:
    logs = sorted(LOGS_DIR.glob("train-it*.log"), key=_iteration_from_name, reverse=True)
    for path in logs:
        lines = path.read_text(errors="replace").splitlines()
        for index in range(len(lines) - 2, -1, -1):
            if lines[index].strip() != TRAIN_RESULT_MARKER:
                continue
            try:
                result = json.loads(lines[index + 1])
            except (IndexError, json.JSONDecodeError):
                break
            result["log"] = str(path)
            return result
    return None


def _gpu_status(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    training = config["training"]
    training_gpu_ids = training.get("gpu_ids")
    if training_gpu_ids is None:
        training_gpu_ids = [training["gpu_id"]]
    selected = sorted({
        *[int(x) for x in config["collection"]["gpu_ids"]],
        *[int(x) for x in training_gpu_ids],
        *[int(x) for x in config["arena"]["gpu_ids"]],
    })
    try:
        inventory = gpu_inventory()
        processes = gpu_compute_processes()
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    current_user = getpass.getuser()
    uuid_to_processes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for process in processes:
        uuid_to_processes[process["gpu_uuid"]].append(process)
    rows = []
    for gpu in inventory:
        if int(gpu["index"]) not in selected:
            continue
        attached = uuid_to_processes[gpu["uuid"]]
        autogo = sum("/envs/autogo/" in str(item["process_name"]) for item in attached)
        other = sum(
            "/envs/autogo/" not in str(item["process_name"])
            or item.get("owner") != current_user
            for item in attached
        )
        rows.append({**gpu, "autogo_processes": autogo, "other_processes": other})
    return rows, None


def build_payload(tail_lines: int = 3) -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    config = load_config()
    stage_scan = _current_stage_scan(state, config)
    npz_count = sum(1 for _ in EXP_DATA_ROOT.rglob("*.npz")) if EXP_DATA_ROOT.exists() else 0
    checkpoints = sorted(CHECKPOINT_ROOT.glob("*.pt")) if CHECKPOINT_ROOT.exists() else []
    newest_log = max(LOGS_DIR.rglob("*.log"), key=lambda p: p.stat().st_mtime, default=None)
    controller_pid = int(state.get("controller_pid") or 0)
    actual_games = int(stage_scan["games"])
    actual_positions = int(stage_scan["positions"])
    gpu_rows, gpu_error = _gpu_status(config)
    return {
        "experiment": EXP_NAME,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": state.get("stage", "NOT_INITIALIZED"),
        "iteration": state.get("iteration"),
        "stage_age_seconds": _timestamp_age(state.get("stage_started_at")),
        "experiment_age_seconds": _timestamp_age(state.get("created_at")),
        "state_age_seconds": _timestamp_age(state.get("updated_at")),
        "champion_iteration": state.get("champion_iteration"),
        "champion_checkpoint": state.get("champion_checkpoint"),
        "candidate_checkpoint": state.get("candidate_checkpoint"),
        "expected_games_current_stage": state.get("expected_games", 0),
        "completed_games_current_stage": actual_games,
        "completed_positions_current_stage": actual_positions,
        "state_completed_games_current_stage": state.get("completed_games", 0),
        "state_completed_positions_current_stage": state.get("completed_positions", 0),
        "current_data": stage_scan,
        "total_npz_files": npz_count,
        "checkpoint_count": len(checkpoints),
        "newest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "controller_pid": controller_pid or None,
        "controller_alive": _pid_alive(controller_pid),
        "child_pids": state.get("child_pids", []),
        "stop_requested": STOP_PATH.exists(),
        "last_error": state.get("last_error"),
        "recovery_command": state.get("recovery_command"),
        "last_arena": state.get("last_arena"),
        "last_promotion": state.get("last_promotion"),
        "arena_history": _arena_history(),
        "latest_training": _latest_training_result(),
        "gpu_rows": gpu_rows,
        "gpu_error": gpu_error,
        "disk_free_gib": round(disk_free_gib(), 1),
        "newest_log": str(newest_log) if newest_log else None,
        "newest_log_age_seconds": (
            round(time.time() - newest_log.stat().st_mtime, 1) if newest_log else None
        ),
        "newest_log_tail": _tail(newest_log, tail_lines) if newest_log else [],
        "events_tail": _tail(EVENTS_PATH, tail_lines),
    }


def _fit(value: Any, width: int) -> str:
    text = " ".join(str(value).splitlines())
    if len(text) <= width:
        return text.ljust(width)
    if width <= 3:
        return text[:width]
    return (text[: width - 3] + "...").ljust(width)


def _table_row(values: Iterable[Any], widths: Iterable[int], aligns: str) -> str:
    cells: list[str] = []
    for value, width, align in zip(values, widths, aligns):
        text = " ".join(str(value).splitlines())
        if len(text) > width:
            text = text[: width - 3] + "..."
        cells.append(f"{text:{align}{width}}")
    return " | ".join(cells)


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    fraction = min(1.0, max(0.0, done / total)) if total else 0.0
    filled = round(fraction * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class Dashboard:
    def __init__(self, width: int) -> None:
        self.width = max(100, width)
        self.inner = self.width - 4
        self.lines: list[str] = []

    def border(self, title: str = "", fill: str = "-") -> None:
        middle = f" {title} " if title else ""
        self.lines.append("+" + middle.center(self.width - 2, fill) + "+")

    def row(self, text: str = "") -> None:
        self.lines.append("| " + _fit(text, self.inner) + " |")

    def render(self) -> str:
        return "\n".join(self.lines)


def render_dashboard(payload: dict[str, Any], width: int) -> str:
    dashboard = Dashboard(width)
    dashboard.border("AutoGo Training Monitor", "=")
    dashboard.row(f"Experiment  {payload['experiment']}")
    dashboard.row(
        f"Updated     {payload['generated_at']}    Experiment age {_human_duration(payload['experiment_age_seconds'])}"
    )

    dashboard.border("LIVE", "-")
    controller = "UP" if payload["controller_alive"] else "DOWN"
    dashboard.row(
        f"Stage {payload['stage']:<20} Iter {int(payload['iteration'] or 0):04d}    "
        f"Champion {int(payload['champion_iteration'] or 0):04d}    "
        f"Controller {controller} pid={payload['controller_pid']} children={len(payload['child_pids'])}"
    )
    done = int(payload["completed_games_current_stage"])
    total = int(payload["expected_games_current_stage"] or 0)
    percent = 100.0 * done / total if total else 0.0
    dashboard.row(
        f"Progress {_progress_bar(done, total)} {done:>3}/{total:<3} {percent:5.1f}%    "
        f"positions={int(payload['completed_positions_current_stage']):,}    "
        f"source={payload['current_data'].get('name') or 'n/a'}"
    )
    dashboard.row(
        f"Stage age {_human_duration(payload['stage_age_seconds']):<10}  "
        f"State age {_human_duration(payload['state_age_seconds']):<10}  "
        f"Stop={'YES' if payload['stop_requested'] else 'no'}  "
        f"Invalid current NPZ={payload['current_data']['invalid']}"
    )
    if payload["last_error"]:
        dashboard.row(f"ERROR       {payload['last_error']}")
        dashboard.row(f"Recovery    {payload['recovery_command']}")

    dashboard.border("CURRENT DATA BREAKDOWN", "-")
    groups = payload["current_data"]["groups"]
    if groups:
        widths = (36, 7, 11, 8, 22, 9, 4)
        aligns = "<>>>>>>"
        dashboard.row(_table_row(
            (
                "Matchup / side",
                "Games",
                "Positions",
                "Avg ply",
                "D4 uniq 4/8/12/20",
                "D4-20 max",
                "Bad",
            ),
            widths,
            aligns,
        ))
        for group in groups:
            games = int(group["games"])
            positions = int(group["positions"])
            average = positions / games if games else 0.0
            eligible = int(group.get("d4_opening_eligible", 0))
            by_depth = group.get("d4_opening_by_depth", {})
            unique_profile = "/".join(
                str(int(by_depth.get(str(depth), {}).get("unique", 0)))
                for depth in D4_OPENING_PREFIX_DEPTHS
            )
            largest_rate = float(group.get("d4_opening_largest_cluster_rate", 0.0))
            dashboard.row(
                _table_row(
                    (
                        group["name"],
                        games,
                        f"{positions:,}",
                        f"{average:.1f}",
                        unique_profile if eligible else "n/a",
                        f"{largest_rate:.1%}" if eligible else "n/a",
                        int(group["invalid"]),
                    ),
                    widths,
                    aligns,
                )
            )
        dashboard.row(
            "D4: unique canonical prefixes at plies 4/8/12/20 after "
            "rotation/reflection; max = D4-20 largest cluster rate."
        )
    else:
        dashboard.row("No completed NPZ files in the current stage yet.")

    arena_live = payload["current_data"].get("arena_live")
    if arena_live is not None:
        dashboard.border("CURRENT ARENA", "-")
        score = arena_live.get("score")
        score_text = f"{float(score):.1%}" if score is not None else "n/a"
        decision = arena_live.get("gatekeeper_decision")
        decision_text = {
            "accept": "locked: accept",
            "reject": "locked: reject",
        }.get(decision, "reversible")
        dashboard.row(
            f"Candidate W-L-D {int(arena_live['wins'])}-{int(arena_live['losses'])}-"
            f"{int(arena_live['draws'])}    Score {score_text}    "
            f"Promotion requires >="
            f"{float(arena_live['promotion_threshold']):.1%}    {decision_text}"
        )
        widths = (20, 7, 13, 7)
        aligns = "<>>>"
        dashboard.row(_table_row(("Candidate color", "Games", "W-L-D", "Score"), widths, aligns))
        for label, key in (
            ("Black", "candidate_as_black"),
            ("White", "candidate_as_white"),
        ):
            side = arena_live[key]
            side_score = side.get("score")
            dashboard.row(
                _table_row(
                    (
                        label,
                        int(side["games"]),
                        f"{int(side['wins'])}-{int(side['losses'])}-{int(side['draws'])}",
                        f"{float(side_score):.1%}" if side_score is not None else "n/a",
                    ),
                    widths,
                    aligns,
                )
            )

    dashboard.border("LATEST TRAINING", "-")
    training = payload["latest_training"]
    if training:
        dashboard.row(
            f"iter={int(training['iteration']):04d}  steps={int(training['steps_completed']):,}  "
            f"elapsed={_human_duration(float(training['elapsed_seconds']))}  stop={training['stop_reason']}  "
            f"peak_vram={float(training['peak_vram_mib']):.0f} MiB"
        )
        dashboard.row(
            f"loss={float(training['train_loss']):.4f}  "
            f"policy_acc={float(training['train_policy_accuracy']):.2%}  "
            f"value_acc={float(training['train_value_accuracy']):.2%}  "
            f"checkpoint={Path(training['checkpoint']).name}"
        )
    else:
        dashboard.row("No completed training result found.")

    dashboard.border("ARENA HISTORY", "-")
    history = payload["arena_history"][-10:]
    if history:
        widths = (4, 11, 7, 9, 9, 7, 8, 10)
        aligns = ">>>>>>>>"
        dashboard.row(_table_row(
            ("Iter", "W-L-D", "Score", "Black", "White", "Max", "Unique", "Decision"),
            widths,
            aligns,
        ))
        for result in history:
            black = result.get("candidate_as_black", {})
            white = result.get("candidate_as_white", {})
            dashboard.row(
                _table_row(
                    (
                        int(result["iteration"]),
                        f"{int(result['wins'])}-{int(result['losses'])}-{int(result['draws'])}",
                        f"{float(result['score']):.1%}",
                        f"{int(black.get('wins', 0))}-{int(black.get('losses', 0))}",
                        f"{int(white.get('wins', 0))}-{int(white.get('losses', 0))}",
                        f"{float(result.get('max_moves_rate', 0.0)):.1%}",
                        f"{float(result.get('unique_trajectory_rate', 0.0)):.1%}",
                        "PROMOTED" if result.get("promoted") else "REJECTED",
                    ),
                    widths,
                    aligns,
                )
            )
    else:
        dashboard.row("No completed Arena result found.")

    dashboard.border("GPU", "-")
    if payload["gpu_rows"]:
        widths = (3, 24, 6, 20, 8, 7)
        aligns = "><>>>>"
        dashboard.row(_table_row(("GPU", "Model", "Util", "Memory", "AutoGo", "Other"), widths, aligns))
        for gpu in payload["gpu_rows"]:
            dashboard.row(
                _table_row(
                    (
                        int(gpu["index"]),
                        gpu["name"],
                        f"{int(gpu['utilization_percent'])}%",
                        f"{int(gpu['memory_used_mib'])}/{int(gpu['memory_total_mib'])} MiB",
                        int(gpu["autogo_processes"]),
                        int(gpu["other_processes"]),
                    ),
                    widths,
                    aligns,
                )
            )
    else:
        dashboard.row(f"GPU telemetry unavailable: {payload['gpu_error']}")

    dashboard.border("ARTIFACTS & LOG", "-")
    dashboard.row(
        f"NPZ total={int(payload['total_npz_files']):,}    checkpoints={int(payload['checkpoint_count'])}    "
        f"disk_free={float(payload['disk_free_gib']):,.1f} GiB    newest_checkpoint="
        f"{Path(payload['newest_checkpoint']).name if payload['newest_checkpoint'] else 'n/a'}"
    )
    if payload["newest_log"]:
        try:
            log_name = str(Path(payload["newest_log"]).relative_to(LOGS_DIR.parent))
        except ValueError:
            log_name = str(payload["newest_log"])
        dashboard.row(
            f"Newest log age={_human_duration(payload['newest_log_age_seconds'])}  {log_name}"
        )
    for line in payload["newest_log_tail"]:
        dashboard.row(f"> {line}")
    dashboard.border(fill="=")
    return dashboard.render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    payload = build_payload(tail_lines=args.tail)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    terminal_width = shutil.get_terminal_size((132, 40)).columns
    width = args.width or min(160, max(100, terminal_width))
    print(render_dashboard(payload, width))


if __name__ == "__main__":
    main()
