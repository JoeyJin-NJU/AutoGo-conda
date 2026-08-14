#!/usr/bin/env python
"""Minimal browser UI for playing against historical AutoGo checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Keep this auxiliary process away from the CPU resources used by training.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import alpha_go_cpp
import numpy as np
import torch

from alpha_go.agents import PASS
from alpha_go.agents.nn_mcts import CppMCTSAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "2026-07-31_11-39-local-9x9-k7-pcr-001"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / EXPERIMENT_NAME
CHECKPOINT_DIR = REPO_ROOT / "local_data" / "checkpoints" / EXPERIMENT_NAME
DEFAULT_CHECKPOINT = "iter1002-candidate.pt"

EVALUATION_NAME = "2026-08-05_16-27-11-iter0152-vs-katago-b10c128-v128"
EVALUATION_DIR = EXPERIMENT_DIR / "evaluations" / EVALUATION_NAME
REPLAY_ROOT = EVALUATION_DIR / "games" / "rl-vs-katago-b10c128-v128"
REPLAY_SPECS = (
    {
        "id": "autogo-loss",
        "outcome": "loss",
        "path": REPLAY_ROOT
        / "rl-black"
        / "shard00"
        / "2026-08-04_08-38-katago-9x9-br-it4-score-001-br-game0000001.npz",
    },
    {
        "id": "autogo-win",
        "outcome": "win",
        "path": REPLAY_ROOT
        / "rl-white"
        / "shard02"
        / "2026-08-04_08-38-katago-9x9-br-it4-score-001-br-game0000014.npz",
    },
)

with (EXPERIMENT_DIR / "config.json").open(encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

sys.path.insert(0, str(EXPERIMENT_DIR))
from color_model import SignedKomiLeafBatchedNNEvaluator  # noqa: E402

BOARD_SIZE = int(CONFIG["rules"]["board_size"])
KOMI = float(CONFIG["rules"]["komi"])
MAX_MOVES = int(CONFIG["rules"]["max_moves"])
BLACK = int(alpha_go_cpp.GoBoard.BLACK)
WHITE = int(alpha_go_cpp.GoBoard.WHITE)

# Use the experiment's Arena search strength, with deterministic move selection.
NUM_SIMULATIONS = int(CONFIG["arena"]["num_simulations"])
C_PUCT = float(CONFIG["arena"]["c_puct"])
LEAF_BATCH_SIZE = int(CONFIG["arena"]["leaf_batch_size"])
MAX_DEPTH = int(CONFIG["mcts"]["max_depth"])
ANALYSIS_CHUNK_SIMULATIONS = max(64, LEAF_BATCH_SIZE)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def checkpoint_names() -> list[str]:
    """Return selectable checkpoints from the current experiment, newest first."""
    paths = [path for path in CHECKPOINT_DIR.glob("*.pt") if path.is_file()]
    paths.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    names = [path.name for path in paths]
    if DEFAULT_CHECKPOINT in names:
        names.remove(DEFAULT_CHECKPOINT)
        names.insert(0, DEFAULT_CHECKPOINT)
    return names


def checkpoint_path(name: str) -> Path:
    """Resolve a name only if it is one of the selectable checkpoints."""
    if name not in checkpoint_names():
        raise ValueError("checkpoint 不存在")
    return CHECKPOINT_DIR / name


def coordinate(row: int, col: int) -> str:
    columns = "ABCDEFGHJ"
    return f"{columns[col]}{BOARD_SIZE - row}"


def _load_replay(spec: dict[str, Any]) -> dict[str, Any]:
    """Load one frozen Arena game and verify every recorded pre-move board."""
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        board_size = int(data["board_size"])
        komi = float(data["komi"])
        saved_boards = data["boards"].copy()
        recorded_moves = data["moves"].copy()
        num_moves = int(data["num_moves"])
        winner = int(data["winner"])
        tracked_color = int(data["tracked_color_code"])
        metadata = {
            "result": str(data["result"]),
            "termination": str(data["termination"]),
            "game_index": int(data["game_index"]),
            "game_seed": int(data["game_seed"]),
            "black_player": str(data["black_agent"]),
            "white_player": str(data["white_agent"]),
            "tracked_color": str(data["tracked_color"]),
            "autogo_simulations": int(data["autogo_num_simulations"]),
            "katago_visits": int(data["katago_max_visits"]),
            "katago_model": str(data["katago_model_name"]),
        }

    if saved_boards.shape != (num_moves, board_size, board_size):
        raise ValueError(f"invalid boards shape in {path}: {saved_boards.shape}")
    if recorded_moves.shape != (num_moves, 2):
        raise ValueError(f"invalid moves shape in {path}: {recorded_moves.shape}")

    autogo_won = winner == tracked_color
    if autogo_won != (spec["outcome"] == "win"):
        raise ValueError(f"selected replay outcome does not match metadata: {path}")

    board = alpha_go_cpp.GoBoard(board_size, komi)
    positions = [board.to_numpy().astype(int).tolist()]
    moves: list[dict[str, Any]] = []
    empty = int(alpha_go_cpp.GoBoard.EMPTY)

    for ply, (row_value, col_value) in enumerate(recorded_moves, start=1):
        before = board.to_numpy()
        if not np.array_equal(before, saved_boards[ply - 1]):
            raise ValueError(f"replay mismatch before ply {ply} in {path}")

        color = int(board.to_play())
        row = int(row_value)
        col = int(col_value)
        is_pass = row < 0
        legal = board.pass_move() if is_pass else board.play(row, col)
        if not legal:
            raise ValueError(f"illegal recorded move at ply {ply} in {path}")

        after = board.to_numpy()
        captured_rows, captured_cols = np.where((before != empty) & (after == empty))
        captures = [
            {"row": int(captured_row), "col": int(captured_col)}
            for captured_row, captured_col in zip(captured_rows, captured_cols)
        ]
        moves.append(
            {
                "number": ply,
                "color": color,
                "row": None if is_pass else row,
                "col": None if is_pass else col,
                "coordinate": "PASS" if is_pass else coordinate(row, col),
                "is_pass": is_pass,
                "captures": captures,
            }
        )
        positions.append(after.astype(int).tolist())

    return {
        "id": spec["id"],
        "outcome": spec["outcome"],
        "board_size": board_size,
        "komi": komi,
        "num_moves": num_moves,
        "positions": positions,
        "moves": moves,
        **metadata,
    }


@lru_cache(maxsize=1)
def replay_payload() -> dict[str, Any]:
    """Return the two curated, engine-verified iter152 versus KataGo games."""
    return {
        "evaluation": EVALUATION_NAME,
        "games": [_load_replay(spec) for spec in REPLAY_SPECS],
    }


class Game:
    def __init__(self) -> None:
        self.board: Any | None = None
        self.agent: CppMCTSAgent | None = None
        self.human_color = BLACK
        self.checkpoint = ""
        self.last_move: tuple[int, int] | None = None
        self._operation_lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        self._analysis_generation = 0
        self._analysis_thread: threading.Thread | None = None
        self._analysis_stop: threading.Event | None = None
        self._analysis_snapshot = self._empty_analysis_snapshot()

    def _empty_analysis_snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self._analysis_generation,
            "status": "idle",
            "running": False,
            "error": None,
            "root_visits": 0,
            "tree_size": 0,
            "visits_per_second": 0.0,
            "elapsed_seconds": 0.0,
            "root_winrate": None,
            "to_play": None,
            "position_move_count": None,
            "checkpoint": self.checkpoint or None,
            "candidates": [],
        }

    def close(self) -> None:
        with self._operation_lock:
            self._stop_analysis_locked(clear=True)
            if self.agent is not None:
                self.agent.close()
            self.agent = None
            self.board = None

    def start(self, checkpoint: str, color: str) -> dict[str, Any]:
        with self._operation_lock:
            if color not in {"black", "white"}:
                raise ValueError("执子方必须是 black 或 white")

            path = checkpoint_path(checkpoint)
            self._stop_analysis_locked(clear=True)
            if self.agent is not None:
                self.agent.close()
            self.agent = None
            self.board = None

            evaluator = SignedKomiLeafBatchedNNEvaluator(
                checkpoint_path=path,
                board_size=BOARD_SIZE,
                komi=KOMI,
                model_config=CONFIG["model"],
                policy_temperature=float(CONFIG["mcts"]["policy_temperature"]),
            )
            self.agent = CppMCTSAgent(
                evaluator=evaluator,
                num_simulations=NUM_SIMULATIONS,
                c_puct=C_PUCT,
                temperature=0.0,
                add_noise=False,
                lambda_=0.0,
                rollout_temperature=1.0,
                max_depth=MAX_DEPTH,
                resign_threshold=0.0,
                leaf_batch_size=LEAF_BATCH_SIZE,
            )
            self.agent.start_game(BOARD_SIZE)
            self.board = alpha_go_cpp.GoBoard(BOARD_SIZE, KOMI)
            self.human_color = BLACK if color == "black" else WHITE
            self.checkpoint = checkpoint
            self.last_move = None

            if self.human_color == WHITE:
                ai_text = self._ai_move()
                return self.state(f"对局开始。{ai_text}")
            return self.state("对局开始。请执黑落子。")

    def play(self, row: int, col: int) -> dict[str, Any]:
        with self._operation_lock:
            self._stop_analysis_locked(clear=True)
            self._require_human_turn()
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                raise ValueError("落子位置超出棋盘")
            if not self.board.is_legal(row, col):
                raise ValueError("该位置不能落子")

            self.board.play(row, col)
            self.last_move = (row, col)
            human_text = f"你在 {coordinate(row, col)} 落子。"
            if self._is_over():
                return self.state(human_text)
            return self.state(f"{human_text}{self._ai_move()}")

    def pass_move(self) -> dict[str, Any]:
        with self._operation_lock:
            self._stop_analysis_locked(clear=True)
            self._require_human_turn()
            self.board.pass_move()
            self.last_move = None
            if self._is_over():
                return self.state("你停一手。")
            return self.state(f"你停一手。{self._ai_move()}")

    def current_state(self) -> dict[str, Any] | None:
        with self._operation_lock:
            if self.board is None:
                return None
            return self.state("已恢复当前对局。")

    def start_analysis(self) -> dict[str, Any]:
        """Continuously add simulations to one tree until explicitly stopped."""
        with self._operation_lock:
            if self.board is None or self.agent is None:
                raise ValueError("请先开始新对局")
            if self._is_over():
                raise ValueError("对局已经结束")

            self._stop_analysis_locked(clear=False)
            board_copy = self.board.copy()
            evaluator = self.agent.evaluator
            cpp_config = self.agent.cpp_config
            to_play = int(self.board.to_play())
            move_count = int(self.board.move_count())

            self._analysis_generation += 1
            session_id = self._analysis_generation
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._analysis_loop,
                args=(
                    session_id,
                    stop_event,
                    board_copy,
                    evaluator,
                    cpp_config,
                    to_play,
                    move_count,
                    self.checkpoint,
                ),
                name=f"mcts-analysis-{session_id}",
                daemon=True,
            )
            with self._analysis_lock:
                self._analysis_stop = stop_event
                self._analysis_thread = thread
                self._analysis_snapshot = {
                    "session_id": session_id,
                    "status": "starting",
                    "running": True,
                    "error": None,
                    "root_visits": 0,
                    "tree_size": 1,
                    "visits_per_second": 0.0,
                    "elapsed_seconds": 0.0,
                    "root_winrate": None,
                    "to_play": to_play,
                    "position_move_count": move_count,
                    "checkpoint": self.checkpoint,
                    "candidates": [],
                }
            thread.start()
            return self.analysis_snapshot()

    def stop_analysis(self) -> dict[str, Any]:
        with self._operation_lock:
            self._stop_analysis_locked(clear=False)
            return self.analysis_snapshot()

    def analysis_snapshot(self) -> dict[str, Any]:
        with self._analysis_lock:
            snapshot = dict(self._analysis_snapshot)
            snapshot["candidates"] = list(self._analysis_snapshot["candidates"])
            return snapshot

    def _stop_analysis_locked(self, *, clear: bool) -> None:
        with self._analysis_lock:
            thread = self._analysis_thread
            stop_event = self._analysis_stop
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("MCTS 分析未能在 30 秒内停止")

        with self._analysis_lock:
            if self._analysis_thread is thread:
                self._analysis_thread = None
                self._analysis_stop = None
            if clear:
                self._analysis_generation += 1
                self._analysis_snapshot = self._empty_analysis_snapshot()
            elif self._analysis_snapshot["running"]:
                self._analysis_snapshot = {
                    **self._analysis_snapshot,
                    "status": "stopped",
                    "running": False,
                }

    def _analysis_loop(
        self,
        session_id: int,
        stop_event: threading.Event,
        board: Any,
        evaluator: Any,
        cpp_config: Any,
        to_play: int,
        move_count: int,
        checkpoint: str,
    ) -> None:
        started = time.monotonic()
        failed = False
        try:
            tree = alpha_go_cpp.MCTSTree(board, cpp_config)
            while not stop_event.is_set():
                if LEAF_BATCH_SIZE > 0 and hasattr(evaluator, "batch_evaluate"):
                    tree.run_simulations_batched(
                        ANALYSIS_CHUNK_SIMULATIONS,
                        LEAF_BATCH_SIZE,
                        evaluator.batch_evaluate,
                    )
                else:
                    tree.run_simulations(
                        ANALYSIS_CHUNK_SIMULATIONS,
                        evaluator.evaluate,
                    )
                snapshot = self._tree_snapshot(
                    session_id=session_id,
                    tree=tree,
                    started=started,
                    to_play=to_play,
                    move_count=move_count,
                    checkpoint=checkpoint,
                )
                with self._analysis_lock:
                    if session_id != self._analysis_generation:
                        return
                    self._analysis_snapshot = snapshot
        except Exception as error:
            failed = True
            with self._analysis_lock:
                if session_id == self._analysis_generation:
                    self._analysis_snapshot = {
                        **self._analysis_snapshot,
                        "status": "error",
                        "running": False,
                        "error": str(error),
                    }
        finally:
            with self._analysis_lock:
                if session_id == self._analysis_generation:
                    if not failed:
                        self._analysis_snapshot = {
                            **self._analysis_snapshot,
                            "status": "stopped",
                            "running": False,
                        }
                    self._analysis_thread = None
                    self._analysis_stop = None

    def _tree_snapshot(
        self,
        *,
        session_id: int,
        tree: Any,
        started: float,
        to_play: int,
        move_count: int,
        checkpoint: str,
    ) -> dict[str, Any]:
        visits = {
            int(action): int(count)
            for action, count in tree.get_child_visit_counts().items()
        }
        q_values = {
            int(action): float(value)
            for action, value in tree.get_child_q_values().items()
        }
        priors = {
            int(action): float(value)
            for action, value in tree.get_root_policy_priors().items()
        }
        root_visits = int(tree.get_root_visit_count())
        child_visits = sum(visits.values())
        elapsed = max(time.monotonic() - started, 1e-9)

        candidates: list[dict[str, Any]] = []
        ordered = sorted(
            visits.items(),
            key=lambda item: (item[1], priors.get(item[0], 0.0)),
            reverse=True,
        )
        for rank, (action, count) in enumerate(ordered, start=1):
            if count <= 0:
                continue
            is_pass = action == int(alpha_go_cpp.PASS_ACTION)
            row = None if is_pass else action // BOARD_SIZE
            col = None if is_pass else action % BOARD_SIZE
            candidates.append(
                {
                    "rank": rank,
                    "action": action,
                    "row": row,
                    "col": col,
                    "coordinate": "PASS" if is_pass else coordinate(row, col),
                    "is_pass": is_pass,
                    "visits": count,
                    "probability": count / child_visits if child_visits else 0.0,
                    "winrate": q_values.get(action),
                    "prior": priors.get(action, 0.0),
                }
            )

        return {
            "session_id": session_id,
            "status": "running",
            "running": True,
            "error": None,
            "root_visits": root_visits,
            "tree_size": int(tree.tree_size()),
            "visits_per_second": root_visits / elapsed,
            "elapsed_seconds": elapsed,
            # Root Q is stored from the previous/opponent player's perspective.
            "root_winrate": 1.0 - float(tree.get_root_q_value()),
            "to_play": to_play,
            "position_move_count": move_count,
            "checkpoint": checkpoint,
            "candidates": candidates,
        }

    def _require_human_turn(self) -> None:
        if self.board is None or self.agent is None:
            raise ValueError("请先开始新对局")
        if self._is_over():
            raise ValueError("对局已经结束")
        if int(self.board.to_play()) != self.human_color:
            raise ValueError("现在不是你的回合")

    def _ai_move(self) -> str:
        if self.board is None or self.agent is None:
            raise RuntimeError("对局尚未初始化")
        move = self.agent.select_move(self.board, seed=int(self.board.move_count()))
        if move == PASS:
            self.board.pass_move()
            self.last_move = None
            return "模型停一手。"

        row, col = int(move[0]), int(move[1])
        if not self.board.play(row, col):
            raise RuntimeError("模型产生了非法落子")
        self.last_move = (row, col)
        return f"模型在 {coordinate(row, col)} 落子。"

    def _is_over(self) -> bool:
        return bool(
            self.board is not None
            and (self.board.is_game_over() or self.board.move_count() >= MAX_MOVES)
        )

    def _result(self) -> str | None:
        if not self._is_over():
            return None
        score = float(self.board.score())
        if score > 0:
            return f"黑胜 {score:.1f} 目"
        if score < 0:
            return f"白胜 {-score:.1f} 目"
        return "和棋"

    def state(self, message: str) -> dict[str, Any]:
        if self.board is None:
            raise RuntimeError("对局尚未初始化")
        result = self._result()
        if result is not None:
            message = f"{message} 对局结束：{result}"
        return {
            "board": self.board.to_numpy().tolist(),
            "to_play": int(self.board.to_play()),
            "human_color": self.human_color,
            "last_move": self.last_move,
            "move_count": int(self.board.move_count()),
            "is_over": result is not None,
            "result": result,
            "checkpoint": self.checkpoint,
            "komi": KOMI,
            "message": message,
        }


GAME = Game()


STATIC_DIR = Path(__file__).resolve().parent
STATIC_ASSETS = {
    "/": ("text/html; charset=utf-8", (STATIC_DIR / "index.html").read_bytes()),
    "/ppt": (
        "text/html; charset=utf-8",
        (STATIC_DIR / "ppt.html").read_bytes(),
    ),
    "/ppt/": (
        "text/html; charset=utf-8",
        (STATIC_DIR / "ppt.html").read_bytes(),
    ),
    "/replay": (
        "text/html; charset=utf-8",
        (STATIC_DIR / "replay.html").read_bytes(),
    ),
    "/replay/": (
        "text/html; charset=utf-8",
        (STATIC_DIR / "replay.html").read_bytes(),
    ),
    "/style.css": ("text/css; charset=utf-8", (STATIC_DIR / "style.css").read_bytes()),
    "/ppt.css": ("text/css; charset=utf-8", (STATIC_DIR / "ppt.css").read_bytes()),
    "/app.js": ("text/javascript; charset=utf-8", (STATIC_DIR / "app.js").read_bytes()),
    "/replay.css": (
        "text/css; charset=utf-8",
        (STATIC_DIR / "replay.css").read_bytes(),
    ),
    "/replay.js": (
        "text/javascript; charset=utf-8",
        (STATIC_DIR / "replay.js").read_bytes(),
    ),
}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, content_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in STATIC_ASSETS:
            self._send_static(*STATIC_ASSETS[path])
        elif path == "/api/checkpoints":
            self._send_json({"checkpoints": checkpoint_names()})
        elif path == "/api/state":
            self._send_json({"state": GAME.current_state()})
        elif path == "/api/analysis":
            self._send_json(GAME.analysis_snapshot())
        elif path == "/api/replays":
            try:
                self._send_json(replay_payload())
            except Exception as error:
                self._send_json({"error": str(error)}, 500)
        else:
            self._send_json({"error": "未找到"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/new-game":
                state = GAME.start(
                    str(payload.get("checkpoint", "")),
                    str(payload.get("color", "")),
                )
            elif path == "/api/move":
                state = GAME.play(int(payload["row"]), int(payload["col"]))
            elif path == "/api/pass":
                state = GAME.pass_move()
            elif path == "/api/analysis/start":
                state = GAME.start_analysis()
            elif path == "/api/analysis/stop":
                state = GAME.stop_analysis()
            else:
                self._send_json({"error": "未找到"}, 404)
                return
            self._send_json(state)
        except (KeyError, TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:
            self._send_json({"error": str(error)}, 500)

    def log_message(self, format: str, *args: Any) -> None:
        if self.command == "GET" and urlsplit(self.path).path == "/api/analysis":
            return
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIR)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"打开 http://{args.host}:{args.port}")
    print(f"checkpoint 目录：{CHECKPOINT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        GAME.close()
        server.server_close()


if __name__ == "__main__":
    main()
