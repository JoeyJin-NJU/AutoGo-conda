"use strict";

const BLACK = 1;
const WHITE = 2;
const COLUMNS = "ABCDEFGHJ";

const elements = {
  loading: document.querySelector("#loading-state"),
  error: document.querySelector("#error-state"),
  errorMessage: document.querySelector("#error-message"),
  retry: document.querySelector("#retry-button"),
  app: document.querySelector("#replay-app"),
  tabs: document.querySelector("#game-tabs"),
  canvas: document.querySelector("#go-board"),
  blackPlayerName: document.querySelector("#black-player-name"),
  whitePlayerName: document.querySelector("#white-player-name"),
  gameContext: document.querySelector("#game-context"),
  gameResult: document.querySelector("#game-result"),
  outcomeTitle: document.querySelector("#outcome-title"),
  outcomeBadge: document.querySelector("#outcome-badge"),
  fraction: document.querySelector("#move-fraction"),
  coordinate: document.querySelector("#move-coordinate"),
  detail: document.querySelector("#move-detail"),
  slider: document.querySelector("#move-slider"),
  percent: document.querySelector("#timeline-percent"),
  first: document.querySelector("#first-button"),
  previous: document.querySelector("#previous-button"),
  play: document.querySelector("#play-button"),
  next: document.querySelector("#next-button"),
  last: document.querySelector("#last-button"),
  speed: document.querySelector("#speed-select"),
  numbers: document.querySelector("#numbers-toggle"),
  moveList: document.querySelector("#move-list"),
  lastMoveCaption: document.querySelector("#last-move-caption"),
  rulesCaption: document.querySelector("#rules-caption"),
  evaluation: document.querySelector("#evaluation-name"),
};

const state = {
  games: [],
  gameIndex: 0,
  ply: 0,
  playing: false,
  timer: null,
};

function currentGame() {
  return state.games[state.gameIndex];
}

function colorName(color) {
  return color === BLACK ? "黑" : "白";
}

function compactPlayer(name) {
  if (name === "RL-iter0152") return "AutoGo · iter152";
  if (name.startsWith("KataGo-")) return "KataGo · b10c128";
  return name;
}

function outcomeCopy(game) {
  if (game.outcome === "win") {
    return {
      tabTitle: "击败 KataGo",
      title: "iter152 半目险胜",
      badge: "AUTOGO WIN",
      color: "执白",
    };
  }
  return {
    tabTitle: "被 KataGo 击败",
    title: "iter152 执黑告负",
    badge: "AUTOGO LOSS",
    color: "执黑",
  };
}

function stopPlayback() {
  if (state.timer !== null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  state.playing = false;
  elements.play.classList.remove("is-playing");
  elements.play.setAttribute("aria-label", "播放");
  elements.play.title = "播放";
}

function scheduleNextMove() {
  if (!state.playing) return;
  const game = currentGame();
  if (state.ply >= game.num_moves) {
    stopPlayback();
    renderControls();
    return;
  }
  const speed = Number(elements.speed.value) || 1;
  state.timer = window.setTimeout(() => {
    state.ply += 1;
    render();
    scheduleNextMove();
  }, 900 / speed);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    renderControls();
    return;
  }
  const game = currentGame();
  if (state.ply >= game.num_moves) state.ply = 0;
  state.playing = true;
  elements.play.classList.add("is-playing");
  elements.play.setAttribute("aria-label", "暂停");
  elements.play.title = "暂停";
  render();
  scheduleNextMove();
}

function setPly(ply) {
  const game = currentGame();
  state.ply = Math.max(0, Math.min(game.num_moves, Number(ply)));
  render();
}

function stoneMoveNumbers(game, ply) {
  const labels = Array.from({ length: game.board_size }, () =>
    Array(game.board_size).fill(null),
  );
  for (let index = 0; index < ply; index += 1) {
    const move = game.moves[index];
    if (!move.is_pass) labels[move.row][move.col] = move.number;
    for (const captured of move.captures) {
      labels[captured.row][captured.col] = null;
    }
  }
  return labels;
}

function drawBoard() {
  const game = currentGame();
  if (!game) return;

  const canvas = elements.canvas;
  const cssSize = Math.max(260, Math.round(canvas.getBoundingClientRect().width));
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const targetSize = Math.round(cssSize * pixelRatio);
  if (canvas.width !== targetSize || canvas.height !== targetSize) {
    canvas.width = targetSize;
    canvas.height = targetSize;
  }

  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, cssSize, cssSize);

  const margin = cssSize * 0.088;
  const boardSpan = cssSize - margin * 2;
  const cell = boardSpan / (game.board_size - 1);
  const stoneRadius = cell * 0.43;

  context.fillStyle = "#d3ab68";
  context.fillRect(0, 0, cssSize, cssSize);

  context.strokeStyle = "rgba(51, 41, 28, 0.88)";
  context.lineWidth = Math.max(1, cssSize / 760);
  for (let index = 0; index < game.board_size; index += 1) {
    const offset = margin + index * cell;
    context.beginPath();
    context.moveTo(margin, offset);
    context.lineTo(cssSize - margin, offset);
    context.stroke();
    context.beginPath();
    context.moveTo(offset, margin);
    context.lineTo(offset, cssSize - margin);
    context.stroke();
  }

  context.fillStyle = "#33291c";
  const starRadius = Math.max(2.2, cell * 0.055);
  for (const row of [2, 4, 6]) {
    for (const col of [2, 4, 6]) {
      context.beginPath();
      context.arc(margin + col * cell, margin + row * cell, starRadius, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.fillStyle = "rgba(51, 41, 28, 0.76)";
  context.font = `600 ${Math.max(8, cssSize * 0.016)}px ui-sans-serif, sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  for (let index = 0; index < game.board_size; index += 1) {
    const offset = margin + index * cell;
    context.fillText(COLUMNS[index], offset, margin * 0.43);
    context.fillText(String(game.board_size - index), margin * 0.43, offset);
  }

  const board = game.positions[state.ply];
  const labels = stoneMoveNumbers(game, state.ply);
  for (let row = 0; row < game.board_size; row += 1) {
    for (let col = 0; col < game.board_size; col += 1) {
      const color = board[row][col];
      if (color !== BLACK && color !== WHITE) continue;
      const x = margin + col * cell;
      const y = margin + row * cell;

      context.save();
      context.shadowColor = "rgba(15, 18, 16, 0.42)";
      context.shadowBlur = cell * 0.12;
      context.shadowOffsetY = cell * 0.075;
      const gradient = context.createRadialGradient(
        x - stoneRadius * 0.32,
        y - stoneRadius * 0.38,
        stoneRadius * 0.08,
        x,
        y,
        stoneRadius,
      );
      if (color === BLACK) {
        gradient.addColorStop(0, "#3b413e");
        gradient.addColorStop(1, "#101311");
      } else {
        gradient.addColorStop(0, "#ffffff");
        gradient.addColorStop(1, "#cbd0cc");
      }
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(x, y, stoneRadius, 0, Math.PI * 2);
      context.fill();
      context.restore();

      if (elements.numbers.checked && labels[row][col] !== null) {
        const number = labels[row][col];
        const digits = String(number).length;
        context.fillStyle = color === BLACK ? "#edf1ed" : "#202621";
        context.font = `700 ${cell * (digits >= 3 ? 0.25 : 0.32)}px ui-sans-serif, sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(String(number), x, y + cell * 0.015);
      }
    }
  }

  if (state.ply > 0) {
    const lastMove = game.moves[state.ply - 1];
    if (!lastMove.is_pass) {
      const x = margin + lastMove.col * cell;
      const y = margin + lastMove.row * cell;
      context.strokeStyle = "#8fb586";
      context.lineWidth = Math.max(2, cell * 0.06);
      context.beginPath();
      context.arc(x, y, stoneRadius * 0.46, 0, Math.PI * 2);
      context.stroke();
    }
  }

  canvas.setAttribute(
    "aria-label",
    `${game.board_size}路棋盘，第 ${state.ply} 手，共 ${game.num_moves} 手`,
  );
}

function renderTabs() {
  elements.tabs.innerHTML = state.games
    .map((game, index) => {
      const copy = outcomeCopy(game);
      return `
        <button
          class="game-tab"
          type="button"
          role="tab"
          data-game-index="${index}"
          aria-selected="${index === state.gameIndex}"
        >
          <span class="tab-index">0${index + 1}</span>
          <span class="tab-copy">
            <strong>${copy.tabTitle}</strong>
            <small>iter152 ${copy.color} · 第 ${game.game_index} 局</small>
          </span>
          <span class="tab-result">${game.result}</span>
        </button>`;
    })
    .join("");
}

function renderMoveList() {
  const game = currentGame();
  elements.moveList.innerHTML = game.moves
    .map((move) => {
      const color = move.color === BLACK ? "black" : "white";
      const captureText = move.captures.length ? `提 ${move.captures.length} 子` : "";
      return `
        <button
          class="move-entry"
          type="button"
          data-ply="${move.number}"
          aria-current="${move.number === state.ply ? "step" : "false"}"
          aria-label="第 ${move.number} 手，${colorName(move.color)}棋 ${move.coordinate}"
        >
          <span class="move-number">${String(move.number).padStart(3, "0")}</span>
          <span class="move-color">
            <i class="mini-stone ${color}" aria-hidden="true"></i>
            ${colorName(move.color)} · ${move.coordinate}
          </span>
          <span class="move-capture">${captureText}</span>
        </button>`;
    })
    .join("");
}

function renderGameMetadata() {
  const game = currentGame();
  const copy = outcomeCopy(game);
  elements.blackPlayerName.textContent = compactPlayer(game.black_player);
  elements.whitePlayerName.textContent = compactPlayer(game.white_player);
  elements.gameContext.textContent = `GAME ${String(game.game_index).padStart(2, "0")}`;
  elements.gameResult.textContent = game.result;
  elements.outcomeTitle.textContent = copy.title;
  elements.outcomeBadge.textContent = copy.badge;
  elements.outcomeBadge.className = `outcome-badge ${game.outcome}`;
  elements.slider.max = String(game.num_moves);
  elements.rulesCaption.textContent = `贴 ${game.komi} 目 · 双方停一手结束`;
}

function renderControls() {
  const game = currentGame();
  const move = state.ply > 0 ? game.moves[state.ply - 1] : null;
  elements.fraction.textContent = `${state.ply} / ${game.num_moves}`;
  elements.slider.value = String(state.ply);
  elements.percent.textContent = `${Math.round((state.ply / game.num_moves) * 100)}%`;
  elements.first.disabled = state.ply === 0;
  elements.previous.disabled = state.ply === 0;
  elements.next.disabled = state.ply === game.num_moves;
  elements.last.disabled = state.ply === game.num_moves;

  if (!move) {
    elements.coordinate.textContent = "开局";
    elements.detail.textContent = "棋盘为空";
    elements.lastMoveCaption.textContent = "初始局面";
  } else {
    elements.coordinate.textContent = move.coordinate;
    const detail = move.is_pass
      ? `${colorName(move.color)}棋停一手`
      : `${colorName(move.color)}棋 · 第 ${move.number} 手${
          move.captures.length ? ` · 提 ${move.captures.length} 子` : ""
        }`;
    elements.detail.textContent = detail;
    elements.lastMoveCaption.textContent = `最后一手：${colorName(move.color)} ${move.coordinate}`;
  }

  const previousCurrent = elements.moveList.querySelector('[aria-current="step"]');
  if (previousCurrent) previousCurrent.setAttribute("aria-current", "false");
  const current = elements.moveList.querySelector(`[data-ply="${state.ply}"]`);
  if (current) {
    current.setAttribute("aria-current", "step");
    current.scrollIntoView({ block: "nearest" });
  }
}

function render() {
  drawBoard();
  renderControls();
}

function selectGame(index) {
  stopPlayback();
  state.gameIndex = index;
  state.ply = 0;
  renderTabs();
  renderGameMetadata();
  renderMoveList();
  render();
}

function showError(message) {
  elements.loading.hidden = true;
  elements.app.hidden = true;
  elements.error.hidden = false;
  elements.errorMessage.textContent = message;
}

async function loadReplays() {
  stopPlayback();
  elements.error.hidden = true;
  elements.app.hidden = true;
  elements.loading.hidden = false;
  try {
    const response = await fetch("/api/replays", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (!Array.isArray(payload.games) || payload.games.length !== 2) {
      throw new Error("回放数据不完整");
    }
    state.games = payload.games;
    elements.evaluation.textContent = payload.evaluation;
    elements.loading.hidden = true;
    elements.app.hidden = false;
    selectGame(0);
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

elements.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-game-index]");
  if (!button) return;
  selectGame(Number(button.dataset.gameIndex));
});

elements.moveList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-ply]");
  if (!button) return;
  stopPlayback();
  setPly(Number(button.dataset.ply));
});

elements.slider.addEventListener("input", () => {
  stopPlayback();
  setPly(elements.slider.value);
});

elements.first.addEventListener("click", () => {
  stopPlayback();
  setPly(0);
});
elements.previous.addEventListener("click", () => {
  stopPlayback();
  setPly(state.ply - 1);
});
elements.play.addEventListener("click", togglePlayback);
elements.next.addEventListener("click", () => {
  stopPlayback();
  setPly(state.ply + 1);
});
elements.last.addEventListener("click", () => {
  stopPlayback();
  setPly(currentGame().num_moves);
});
elements.numbers.addEventListener("change", drawBoard);
elements.retry.addEventListener("click", loadReplays);

document.addEventListener("keydown", (event) => {
  if (!state.games.length) return;
  const target = event.target;
  if (target instanceof HTMLSelectElement || target instanceof HTMLInputElement) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    stopPlayback();
    setPly(state.ply - 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    stopPlayback();
    setPly(state.ply + 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    stopPlayback();
    setPly(0);
  } else if (event.key === "End") {
    event.preventDefault();
    stopPlayback();
    setPly(currentGame().num_moves);
  } else if (event.key === " ") {
    event.preventDefault();
    togglePlayback();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopPlayback();
    if (state.games.length) renderControls();
  }
});

window.addEventListener("resize", drawBoard);
window.addEventListener("beforeunload", stopPlayback);

loadReplays();
