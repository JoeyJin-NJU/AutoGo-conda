"""FastAPI web server for playing Go using GNU Go as the engine."""
from __future__ import annotations

import argparse
import io
import uuid
from pathlib import Path

import alpha_go_cpp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from alpha_go.engine import BLACK, COLS, WHITE, GTPEngine

app = FastAPI(title="Go Game")

# Store active games
games: dict[str, GTPEngine] = {}
game_human_color: dict[str, int] = {}
game_agents: dict[str, "Agent"] = {}
game_agent_names: dict[str, str] = {}  # Store agent name for recreation on undo

# Import agents (must be after GTPEngine is defined)
from alpha_go.agents import Agent, get_agent, list_agents  # noqa: E402

# Lazy-loaded KataGo model for assist feature
_assist_model = None
_assist_gs = None  # GameState for assist


def _get_assist_model():
    """Lazy-load KataGo model for assist feature."""
    global _assist_model
    if _assist_model is None:
        import torch
        from katago.train.load_model import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _assist_model, _, _ = load_model(
            "kata1-b28c512nbt-adam-s11165M-d5387M/model.ckpt",
            use_swa=False,
            device=device,
            pos_len=19,
            verbose=False,
        )
        _assist_model.eval()
    return _assist_model


class MoveRequest(BaseModel):
    row: int | None = None
    col: int | None = None
    pass_move: bool = False


class GameState(BaseModel):
    game_id: str
    board: list[list[int]]
    size: int
    to_play: int
    last_move: tuple[int, int] | None
    is_over: bool
    result: str | None
    legal_moves: list[tuple[int, int]]
    human_color: int
    message: str


def engine_to_state(game_id: str, engine: GTPEngine, message: str = "") -> GameState:
    """Convert engine state to API response."""
    return GameState(
        game_id=game_id,
        board=engine.get_board(),
        size=engine.size,
        to_play=engine.to_play,
        last_move=engine.last_move,
        is_over=engine.is_over(),
        result=engine.result(),
        legal_moves=engine.get_legal_moves(),
        human_color=game_human_color.get(game_id, BLACK),
        message=message,
    )


def _engine_to_cpp_board(engine: GTPEngine) -> alpha_go_cpp.GoBoard:
    """Convert GTPEngine state to alpha_go_cpp.GoBoard."""
    cpp_board = alpha_go_cpp.GoBoard(engine.size)
    board_np = np.array(engine.get_board(), dtype=np.int8)
    to_play = alpha_go_cpp.GoBoard.BLACK if engine.to_play == BLACK else alpha_go_cpp.GoBoard.WHITE
    cpp_board.set_from_numpy(board_np, to_play)
    return cpp_board


def _ai_make_move(game_id: str) -> tuple[int, int] | None:
    """Have the AI agent make a move."""
    engine = games[game_id]
    agent = game_agents[game_id]

    # Convert engine state to C++ board for agents that need it
    cpp_board = _engine_to_cpp_board(engine)
    move = agent.select_move(cpp_board, seed=0)

    # Handle PASS (-1, -1) or None
    if move is None or move == (-1, -1):
        engine.play(None, None)
        agent.notify_move(-1, -1)
        return None
    else:
        engine.play(move[0], move[1])
        agent.notify_move(move[0], move[1])
    return move


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the game UI."""
    return HTML_PAGE


@app.get("/api/agents")
async def get_agents() -> list[str]:
    """Get list of available agents."""
    return list_agents()


@app.post("/api/new_game")
async def new_game(
    size: int = 9, color: str = "black", agent: str = "gnugo1"
) -> GameState:
    """Create a new game."""
    if size not in (9, 13, 19):
        raise HTTPException(status_code=400, detail="Size must be 9, 13, or 19")

    try:
        ai_agent = get_agent(agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    game_id = str(uuid.uuid4())[:8]
    engine = GTPEngine.new(size=size, level=1)  # Main engine just tracks state
    games[game_id] = engine
    game_agents[game_id] = ai_agent
    game_agent_names[game_id] = agent  # Store name for recreation on undo
    human_color = BLACK if color == "black" else WHITE
    game_human_color[game_id] = human_color

    # Initialize agent
    ai_agent.start_game(size)

    # If human is white, AI plays first
    if human_color == WHITE:
        _ai_make_move(game_id)

    return engine_to_state(game_id, engine, "Game started!")


@app.get("/api/game/{game_id}")
async def get_game(game_id: str) -> GameState:
    """Get current game state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return engine_to_state(game_id, games[game_id])


@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, move: MoveRequest) -> GameState:
    """Make a move."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    engine = games[game_id]
    human_color = game_human_color[game_id]

    if engine.is_over():
        return engine_to_state(game_id, engine, "Game is already over")

    # Check it's human's turn
    if engine.to_play != human_color:
        return engine_to_state(game_id, engine, "Not your turn!")

    # Play human move
    agent = game_agents[game_id]
    if move.pass_move:
        engine.play(None, None)
        agent.notify_move(None, None)
        msg = "You passed."
    elif move.row is not None and move.col is not None:
        if not engine.is_legal(move.row, move.col):
            return engine_to_state(game_id, engine, "Illegal move!")
        engine.play(move.row, move.col)
        agent.notify_move(move.row, move.col)
        msg = f"You played at {COLS[move.col]}{engine.size - move.row}."
    else:
        return engine_to_state(game_id, engine, "Invalid move format")

    # Check if game ended
    if engine.is_over():
        return engine_to_state(game_id, engine, f"Game over! {engine.result()}")

    # AI responds
    ai_move = _ai_make_move(game_id)
    agent_name = type(agent).__name__
    if ai_move is None:
        msg += f" {agent_name} passed."
    else:
        msg += f" {agent_name} played at {COLS[ai_move[1]]}{engine.size - ai_move[0]}."

    if engine.is_over():
        msg = f"Game over! {engine.result()}"

    return engine_to_state(game_id, engine, msg)


@app.post("/api/game/{game_id}/pass")
async def pass_move(game_id: str) -> GameState:
    """Pass turn."""
    return await make_move(game_id, MoveRequest(pass_move=True))


@app.post("/api/game/{game_id}/undo")
async def undo_move(game_id: str) -> GameState:
    """Undo the last two moves (AI move + human move)."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    engine = games[game_id]
    human_color = game_human_color[game_id]

    # Need at least 2 moves to undo (human + AI)
    if len(engine.move_history) < 2:
        return engine_to_state(game_id, engine, "Not enough moves to undo")

    # Undo AI's last move and human's last move
    if not engine.undo():
        return engine_to_state(game_id, engine, "Failed to undo")
    if not engine.undo():
        return engine_to_state(game_id, engine, "Failed to undo")

    # Recreate agent and replay moves to sync its internal state
    agent = game_agents[game_id]
    agent.end_game()

    # Recreate agent using stored name
    agent_name = game_agent_names[game_id]
    agent = get_agent(agent_name)
    agent.start_game(engine.size)
    game_agents[game_id] = agent

    # Replay all remaining moves to sync agent state
    for move in engine.move_history:
        if move is None:
            agent.notify_move(-1, -1)
        else:
            agent.notify_move(move[0], move[1])

    return engine_to_state(game_id, engine, "Undid last 2 moves")


@app.get("/api/game/{game_id}/assist")
async def get_assist_probabilities(game_id: str) -> dict:
    """Get KataGo move probabilities for assist visualization."""
    import torch
    from katago.game.gamestate import GameState
    from katago.game.board import Board

    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    engine = games[game_id]
    size = engine.size

    # Load model (lazy)
    model = _get_assist_model()

    # Create GameState and replay moves
    gs = GameState(size, GameState.RULES_TT)
    for move in engine.move_history:
        pla = gs.board.pla
        if move is None:
            loc = Board.PASS_LOC
        else:
            loc = gs.board.loc(move[1], move[0])  # KataGo uses (x=col, y=row)
        gs.play(pla, loc)

    # Get model outputs
    with torch.no_grad():
        outputs = gs.get_model_outputs(model)

    # Convert to (row, col) -> probability dict
    probabilities: dict[str, float] = {}
    for loc, prob in outputs["moves_and_probs0"]:
        if loc == Board.PASS_LOC:
            continue  # Skip pass move
        row = gs.board.loc_y(loc)
        col = gs.board.loc_x(loc)
        # Use string key for JSON serialization
        probabilities[f"{row},{col}"] = float(prob)

    return {"probabilities": probabilities}


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up GNU Go processes on shutdown."""
    for agent in game_agents.values():
        agent.end_game()
    for engine in games.values():
        engine.close()


# Replay functionality
replay_data: dict | None = None
replay_files: list[str] = []  # List of all replay files in directory
replay_current_idx: int = 0  # Current replay index


def _load_replay_from_env() -> None:
    """Load replay file(s) from environment variable if set."""
    global replay_data, replay_files, replay_current_idx
    import os
    path = os.environ.get("ALPHA_GO_REPLAY_FILE")
    if path:
        p = Path(path)
        if p.is_dir():
            # Load all npz files in directory
            replay_files = sorted([str(f) for f in p.glob("*.npz")])
            if replay_files:
                replay_current_idx = 0
                replay_data = dict(np.load(replay_files[0], allow_pickle=True))
        elif p.exists():
            # Single file
            replay_files = [str(p)]
            replay_current_idx = 0
            replay_data = dict(np.load(p, allow_pickle=True))


_load_replay_from_env()


class ReplayState(BaseModel):
    board: list[list[int]]
    size: int
    move_number: int
    total_moves: int
    last_move: tuple[int, int] | None
    result: str
    black_agent: str
    white_agent: str


@app.post("/api/replay/upload")
async def upload_replay(file: UploadFile) -> dict:
    """Upload a replay file."""
    global replay_data
    content = await file.read()
    replay_data = dict(np.load(io.BytesIO(content), allow_pickle=True))
    return {"status": "ok"}


@app.get("/api/replay")
async def get_replay_info() -> dict:
    """Get replay metadata."""
    if replay_data is None:
        raise HTTPException(status_code=404, detail="No replay loaded")
    return {
        "size": int(replay_data["board_size"]),
        "total_moves": int(replay_data["num_moves"]),
        "result": str(replay_data["result"]),
        "black_agent": str(replay_data["black_agent"]),
        "white_agent": str(replay_data["white_agent"]),
        "winner": int(replay_data["winner"]),
        "game_index": replay_current_idx,
        "total_games": len(replay_files),
        "filename": Path(replay_files[replay_current_idx]).name if replay_files else "",
    }


@app.post("/api/replay/next")
async def next_replay() -> dict:
    """Load the next replay file."""
    global replay_data, replay_current_idx
    if not replay_files:
        raise HTTPException(status_code=404, detail="No replays loaded")
    if replay_current_idx >= len(replay_files) - 1:
        raise HTTPException(status_code=400, detail="Already at last game")
    replay_current_idx += 1
    replay_data = dict(np.load(replay_files[replay_current_idx], allow_pickle=True))
    return await get_replay_info()


@app.post("/api/replay/prev")
async def prev_replay() -> dict:
    """Load the previous replay file."""
    global replay_data, replay_current_idx
    if not replay_files:
        raise HTTPException(status_code=404, detail="No replays loaded")
    if replay_current_idx <= 0:
        raise HTTPException(status_code=400, detail="Already at first game")
    replay_current_idx -= 1
    replay_data = dict(np.load(replay_files[replay_current_idx], allow_pickle=True))
    return await get_replay_info()


@app.post("/api/replay/goto/{game_index}")
async def goto_replay(game_index: int) -> dict:
    """Load a specific replay by index."""
    global replay_data, replay_current_idx
    if not replay_files:
        raise HTTPException(status_code=404, detail="No replays loaded")
    if game_index < 0 or game_index >= len(replay_files):
        raise HTTPException(status_code=400, detail="Invalid game index")
    replay_current_idx = game_index
    replay_data = dict(np.load(replay_files[replay_current_idx], allow_pickle=True))
    return await get_replay_info()


@app.get("/api/replay/{move_number}")
async def get_replay_state(move_number: int) -> ReplayState:
    """Get board state at a specific move."""
    if replay_data is None:
        raise HTTPException(status_code=404, detail="No replay loaded")

    total_moves = int(replay_data["num_moves"])
    if move_number < 0 or move_number > total_moves:
        raise HTTPException(status_code=400, detail="Invalid move number")

    size = int(replay_data["board_size"])

    if move_number == 0:
        board = [[0] * size for _ in range(size)]
        last_move = None
    else:
        board = replay_data["boards"][move_number - 1].tolist()
        move = replay_data["moves"][move_number - 1]
        if move[0] == -1:
            last_move = None
        else:
            last_move = (int(move[0]), int(move[1]))

    return ReplayState(
        board=board,
        size=size,
        move_number=move_number,
        total_moves=total_moves,
        last_move=last_move,
        result=str(replay_data["result"]),
        black_agent=str(replay_data["black_agent"]),
        white_agent=str(replay_data["white_agent"]),
    )


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Go Game</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; text-align: center; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        select, button {
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button, .replay-btn { background: #4CAF50; color: white; }
        button:hover, .replay-btn:hover { background: #45a049; }
        button.pass { background: #ff9800; }
        button.pass:hover { background: #f57c00; }
        button.undo { background: #9c27b0; }
        button.undo:hover { background: #7b1fa2; }
        .replay-btn {
            background: #2196F3;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .replay-btn:hover { background: #1976D2; }
        select { background: white; border: 1px solid #ddd; }
        .board-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .board {
            background: #DEB887;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            display: inline-block;
        }
        .board-grid {
            display: grid;
            gap: 0;
        }
        .cell {
            width: 36px;
            height: 36px;
            position: relative;
            cursor: pointer;
        }
        .cell::before {
            content: '';
            position: absolute;
            background: #8B4513;
        }
        .cell::before {
            left: 50%;
            top: 0;
            bottom: 0;
            width: 1px;
        }
        .cell::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #8B4513;
        }
        .cell.top::before { top: 50%; }
        .cell.bottom::before { bottom: 50%; }
        .cell.left::after { left: 50%; }
        .cell.right::after { right: 50%; }
        .stone {
            position: absolute;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 10;
        }
        .stone.black {
            background: radial-gradient(circle at 30% 30%, #555, #000);
            box-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .stone.white {
            background: radial-gradient(circle at 30% 30%, #fff, #ccc);
            box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stone.last-move::after {
            content: '';
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        .cell:hover:not(.occupied)::before {
            content: '';
            position: absolute;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 2px dashed #666;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 5;
        }
        .star-point {
            position: absolute;
            width: 8px;
            height: 8px;
            background: #8B4513;
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1;
        }
        .status {
            text-align: center;
            padding: 15px;
            margin: 10px 0;
            background: white;
            border-radius: 5px;
            font-size: 18px;
        }
        .status.your-turn { background: #e8f5e9; }
        .status.waiting { background: #fff3e0; }
        .status.game-over { background: #ffebee; font-weight: bold; }
        .legal-move-marker {
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: rgba(76, 175, 80, 0.6);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 5;
        }
        .assist-marker {
            position: absolute;
            border-radius: 50%;
            background: rgba(76, 175, 80, 0.4);
            border: 1px solid rgba(76, 175, 80, 0.8);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 4;
        }
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .toggle-container label { cursor: pointer; user-select: none; }
        .toggle-switch {
            position: relative;
            width: 40px;
            height: 22px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #ccc;
            transition: .3s;
            border-radius: 22px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        .toggle-switch input:checked + .toggle-slider { background-color: #4CAF50; }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(18px); }
        .message {
            text-align: center;
            color: #666;
            margin: 10px 0;
        }
        .coords {
            display: flex;
            justify-content: center;
            gap: 0;
            margin-left: 20px;
        }
        .coord-label {
            width: 36px;
            text-align: center;
            font-weight: bold;
            color: #666;
        }
        .row-labels {
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-right: 5px;
        }
        .row-label {
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: #666;
            width: 20px;
        }
        /* Replay controls */
        .replay-controls {
            display: none;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 5px;
        }
        .replay-controls.active { display: flex; }
        .replay-controls button {
            padding: 8px 16px;
            font-size: 18px;
            min-width: 40px;
        }
        .slider-container {
            flex: 1;
            min-width: 150px;
            max-width: 300px;
        }
        .slider-container input[type="range"] {
            width: 100%;
            cursor: pointer;
        }
        .move-display {
            min-width: 100px;
            text-align: center;
            font-weight: bold;
        }
        .replay-info {
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: #555;
        }
        .game-nav {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin: 5px 0;
        }
        .game-nav button {
            background: #9c27b0;
            padding: 6px 12px;
            font-size: 14px;
        }
        .game-nav button:hover { background: #7b1fa2; }
        .game-nav button:disabled { background: #ccc; cursor: not-allowed; }
        #gameCounter {
            font-weight: bold;
            min-width: 80px;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Go Game (GNU Go)</h1>

    <div class="controls">
        <select id="size">
            <option value="9">9x9</option>
            <option value="13">13x13</option>
            <option value="19">19x19</option>
        </select>
        <select id="color">
            <option value="black">Play Black</option>
            <option value="white">Play White</option>
        </select>
        <select id="agent"></select>
        <button onclick="newGame()">New Game</button>
        <button class="pass" onclick="pass()">Pass</button>
        <button class="undo" onclick="undo()">Undo</button>
        <label class="replay-btn">
            Load Replay
            <input type="file" id="replayFile" accept=".npz" onchange="uploadReplay(this)" hidden>
        </label>
        <div class="toggle-container">
            <label class="toggle-switch">
                <input type="checkbox" id="showLegalMoves" onchange="renderBoard(gameState)">
                <span class="toggle-slider"></span>
            </label>
            <label for="showLegalMoves">Show Legal Moves</label>
        </div>
        <div class="toggle-container">
            <label class="toggle-switch">
                <input type="checkbox" id="showAssist" onchange="toggleAssist()">
                <span class="toggle-slider"></span>
            </label>
            <label for="showAssist">Show Assist</label>
        </div>
    </div>

    <div class="replay-controls" id="replayControls">
        <div class="replay-info" id="replayInfo"></div>
        <div class="game-nav" id="gameNav" style="display: none;">
            <button onclick="prevGame()" id="prevGameBtn">&laquo; Prev Game</button>
            <span id="gameCounter">Game 1/1</span>
            <button onclick="nextGame()" id="nextGameBtn">Next Game &raquo;</button>
        </div>
        <button onclick="goToStart()">&laquo;</button>
        <button onclick="stepBack()">&lsaquo;</button>
        <div class="slider-container">
            <input type="range" id="replaySlider" min="0" value="0" oninput="goToMove(this.value)">
        </div>
        <button onclick="stepForward()">&rsaquo;</button>
        <button onclick="goToEnd()">&raquo;</button>
        <div class="move-display" id="moveDisplay">Move 0</div>
        <button onclick="exitReplay()" style="background: #f44336;">Exit Replay</button>
    </div>

    <div id="status" class="status">Click "New Game" to start!</div>
    <div id="message" class="message"></div>

    <div class="board-container">
        <div class="row-labels" id="row-labels"></div>
        <div>
            <div class="coords" id="col-labels"></div>
            <div class="board">
                <div class="board-grid" id="board"></div>
            </div>
        </div>
    </div>

    <script>
        let gameId = null;
        let gameState = null;
        let replayMode = false;
        let replayInfo = null;
        let currentMove = 0;
        let assistProbabilities = null;  // {row,col: probability}

        const COLS = 'ABCDEFGHJKLMNOPQRST';  // Skip I

        function getStarPoints(size) {
            if (size === 9) return [[2,2], [2,6], [4,4], [6,2], [6,6]];
            if (size === 13) return [[3,3], [3,9], [6,6], [9,3], [9,9]];
            return [[3,3], [3,9], [3,15], [9,3], [9,9], [9,15], [15,3], [15,9], [15,15]];
        }

        function renderBoard(state) {
            const board = document.getElementById('board');
            const colLabels = document.getElementById('col-labels');
            const rowLabels = document.getElementById('row-labels');
            const size = state.size;

            board.style.gridTemplateColumns = `repeat(${size}, 36px)`;
            board.innerHTML = '';
            colLabels.innerHTML = '';
            rowLabels.innerHTML = '';

            // Column labels
            for (let c = 0; c < size; c++) {
                const label = document.createElement('div');
                label.className = 'coord-label';
                label.textContent = COLS[c];
                colLabels.appendChild(label);
            }

            // Row labels
            for (let r = 0; r < size; r++) {
                const label = document.createElement('div');
                label.className = 'row-label';
                label.textContent = size - r;
                rowLabels.appendChild(label);
            }

            const starPoints = getStarPoints(size);

            for (let r = 0; r < size; r++) {
                for (let c = 0; c < size; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';

                    // Edge classes
                    if (r === 0) cell.classList.add('top');
                    if (r === size - 1) cell.classList.add('bottom');
                    if (c === 0) cell.classList.add('left');
                    if (c === size - 1) cell.classList.add('right');

                    // Star point
                    if (starPoints.some(([sr, sc]) => sr === r && sc === c)) {
                        const star = document.createElement('div');
                        star.className = 'star-point';
                        cell.appendChild(star);
                    }

                    // Stone
                    const stone = state.board[r][c];
                    if (stone !== 0) {
                        const stoneEl = document.createElement('div');
                        stoneEl.className = 'stone ' + (stone === 1 ? 'black' : 'white');
                        if (state.last_move && state.last_move[0] === r && state.last_move[1] === c) {
                            stoneEl.classList.add('last-move');
                        }
                        cell.appendChild(stoneEl);
                        cell.classList.add('occupied');
                    }

                    // Legal move marker
                    const showLegal = document.getElementById('showLegalMoves')?.checked;
                    if (showLegal && state.legal_moves && stone === 0) {
                        const isLegal = state.legal_moves.some(([mr, mc]) => mr === r && mc === c);
                        if (isLegal) {
                            const marker = document.createElement('div');
                            marker.className = 'legal-move-marker';
                            cell.appendChild(marker);
                        }
                    }

                    // Assist probability marker
                    const showAssist = document.getElementById('showAssist')?.checked;
                    if (showAssist && assistProbabilities && stone === 0) {
                        const prob = assistProbabilities[`${r},${c}`];
                        if (prob && prob > 0.01) {  // Only show if > 1%
                            const marker = document.createElement('div');
                            marker.className = 'assist-marker';
                            // Scale radius: sqrt for better visual distribution, max 30px
                            const radius = Math.min(30, Math.max(6, Math.sqrt(prob) * 40));
                            marker.style.width = `${radius}px`;
                            marker.style.height = `${radius}px`;
                            // Add probability text for high-probability moves
                            if (prob > 0.05) {
                                marker.title = `${(prob * 100).toFixed(1)}%`;
                            }
                            cell.appendChild(marker);
                        }
                    }

                    cell.onclick = () => makeMove(r, c);
                    board.appendChild(cell);
                }
            }
        }

        function updateStatus(state) {
            const status = document.getElementById('status');
            const message = document.getElementById('message');

            if (state.is_over) {
                status.className = 'status game-over';
                status.textContent = `Game Over: ${state.result}`;
            } else if (state.to_play === state.human_color) {
                status.className = 'status your-turn';
                status.textContent = state.human_color === 1 ? 'Your turn (Black)' : 'Your turn (White)';
            } else {
                status.className = 'status waiting';
                const agent = document.getElementById('agent').value;
                status.textContent = `${agent} is thinking...`;
            }

            message.textContent = state.message || '';
        }

        async function loadAgents() {
            const response = await fetch('/api/agents');
            const agents = await response.json();
            const select = document.getElementById('agent');
            select.innerHTML = agents.map(a =>
                `<option value="${a}"${a === 'gnugo1' ? ' selected' : ''}>${a}</option>`
            ).join('');
        }

        async function newGame() {
            const size = document.getElementById('size').value;
            const color = document.getElementById('color').value;
            const agent = document.getElementById('agent').value;

            const response = await fetch(`/api/new_game?size=${size}&color=${color}&agent=${agent}`, {
                method: 'POST'
            });
            gameState = await response.json();
            gameId = gameState.game_id;
            assistProbabilities = null;  // Reset assist
            await refreshAssistIfEnabled();
            renderBoard(gameState);
            updateStatus(gameState);
        }

        async function makeMove(row, col) {
            if (!gameId || gameState.is_over) return;
            if (gameState.to_play !== gameState.human_color) return;

            const response = await fetch(`/api/game/${gameId}/move`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({row, col, pass_move: false})
            });
            gameState = await response.json();
            await refreshAssistIfEnabled();
            renderBoard(gameState);
            updateStatus(gameState);
        }

        async function pass() {
            if (!gameId || gameState.is_over) return;
            if (gameState.to_play !== gameState.human_color) return;

            const response = await fetch(`/api/game/${gameId}/pass`, {method: 'POST'});
            gameState = await response.json();
            await refreshAssistIfEnabled();
            renderBoard(gameState);
            updateStatus(gameState);
        }

        async function undo() {
            if (!gameId) return;

            const response = await fetch(`/api/game/${gameId}/undo`, {method: 'POST'});
            gameState = await response.json();
            await refreshAssistIfEnabled();
            renderBoard(gameState);
            updateStatus(gameState);
        }

        async function toggleAssist() {
            const showAssist = document.getElementById('showAssist')?.checked;
            if (showAssist && gameId) {
                await fetchAssistProbabilities();
            } else {
                assistProbabilities = null;
            }
            renderBoard(gameState);
        }

        async function fetchAssistProbabilities() {
            if (!gameId) return;
            try {
                const response = await fetch(`/api/game/${gameId}/assist`);
                if (response.ok) {
                    const data = await response.json();
                    assistProbabilities = data.probabilities;
                }
            } catch (e) {
                console.error('Failed to fetch assist probabilities:', e);
            }
        }

        async function refreshAssistIfEnabled() {
            const showAssist = document.getElementById('showAssist')?.checked;
            if (showAssist && gameId) {
                await fetchAssistProbabilities();
            }
        }

        async function uploadReplay(input) {
            if (!input.files.length) return;
            const formData = new FormData();
            formData.append('file', input.files[0]);
            await fetch('/api/replay/upload', { method: 'POST', body: formData });
            input.value = '';
            await enterReplayMode();
        }

        async function enterReplayMode() {
            const response = await fetch('/api/replay');
            if (!response.ok) return;
            replayInfo = await response.json();
            replayMode = true;

            // Show replay controls, hide game controls
            document.getElementById('replayControls').classList.add('active');
            document.querySelector('.controls').style.display = 'none';
            document.getElementById('status').style.display = 'none';
            document.getElementById('message').style.display = 'none';

            updateReplayInfo();
            await goToMove(0);
        }

        function updateReplayInfo() {
            // Update replay info
            let infoHtml = `
                <strong>${replayInfo.black_agent}</strong> (Black) vs
                <strong>${replayInfo.white_agent}</strong> (White) |
                ${replayInfo.size}x${replayInfo.size} |
                Result: ${replayInfo.result}
            `;
            if (replayInfo.filename) {
                infoHtml += ` | <em>${replayInfo.filename}</em>`;
            }
            document.getElementById('replayInfo').innerHTML = infoHtml;
            document.getElementById('replaySlider').max = replayInfo.total_moves;

            // Show/hide game navigation
            const gameNav = document.getElementById('gameNav');
            if (replayInfo.total_games > 1) {
                gameNav.style.display = 'flex';
                document.getElementById('gameCounter').textContent =
                    `Game ${replayInfo.game_index + 1}/${replayInfo.total_games}`;
                document.getElementById('prevGameBtn').disabled = replayInfo.game_index <= 0;
                document.getElementById('nextGameBtn').disabled = replayInfo.game_index >= replayInfo.total_games - 1;
            } else {
                gameNav.style.display = 'none';
            }
        }

        async function nextGame() {
            const response = await fetch('/api/replay/next', { method: 'POST' });
            if (!response.ok) return;
            replayInfo = await response.json();
            updateReplayInfo();
            await goToMove(0);
        }

        async function prevGame() {
            const response = await fetch('/api/replay/prev', { method: 'POST' });
            if (!response.ok) return;
            replayInfo = await response.json();
            updateReplayInfo();
            await goToMove(0);
        }

        function exitReplay() {
            replayMode = false;
            replayInfo = null;
            currentMove = 0;

            // Hide replay controls, show game controls
            document.getElementById('replayControls').classList.remove('active');
            document.querySelector('.controls').style.display = 'flex';
            document.getElementById('status').style.display = 'block';
            document.getElementById('message').style.display = 'block';

            // Restore game board
            if (gameState) {
                renderBoard(gameState);
                updateStatus(gameState);
            }
        }

        async function goToMove(moveNum) {
            moveNum = parseInt(moveNum);
            currentMove = moveNum;
            const response = await fetch(`/api/replay/${moveNum}`);
            const state = await response.json();
            renderBoard(state);
            document.getElementById('moveDisplay').textContent = `Move ${moveNum}/${replayInfo.total_moves}`;
            document.getElementById('replaySlider').value = moveNum;
        }

        function stepBack() { if (currentMove > 0) goToMove(currentMove - 1); }
        function stepForward() { if (replayInfo && currentMove < replayInfo.total_moves) goToMove(currentMove + 1); }
        function goToStart() { goToMove(0); }
        function goToEnd() { if (replayInfo) goToMove(replayInfo.total_moves); }

        // Load agents and start a new game on page load
        window.onload = async () => {
            await loadAgents();
            // Check if replay is pre-loaded
            try {
                const resp = await fetch('/api/replay');
                if (resp.ok) {
                    await enterReplayMode();
                    return;
                }
            } catch (e) {}
            await newGame();
        };

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+Z for undo in game mode
            if (!replayMode && e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                undo();
                return;
            }

            // Replay mode shortcuts
            if (!replayMode) return;
            if (e.key === 'ArrowLeft') stepBack();
            if (e.key === 'ArrowRight') stepForward();
            if (e.key === 'Home') goToStart();
            if (e.key === 'End') goToEnd();
            if (e.key === 'Escape') exitReplay();
            if (e.key === 'PageUp' || e.key === '[') prevGame();
            if (e.key === 'PageDown' || e.key === ']') nextGame();
        });
    </script>
</body>
</html>
"""


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Run Go game web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--replay", type=str, help="Path to replay file (.npz) or directory of .npz files")
    args = parser.parse_args()

    if args.replay:
        replay_path = Path(args.replay).resolve()
        if not replay_path.exists():
            print(f"Error: Replay path not found: {args.replay}")
            return
        os.environ["ALPHA_GO_REPLAY_FILE"] = str(replay_path)
        if replay_path.is_dir():
            npz_count = len(list(replay_path.glob("*.npz")))
            print(f"Starting server at http://{args.host}:{args.port} (replay mode)")
            print(f"Loaded directory: {args.replay} ({npz_count} games)")
        else:
            print(f"Starting server at http://{args.host}:{args.port} (replay mode)")
            print(f"Loaded: {args.replay}")
    else:
        print(f"Starting Go game server at http://{args.host}:{args.port}")
        print("Using GNU Go as the game engine")

    uvicorn.run(
        "alpha_go.play:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
