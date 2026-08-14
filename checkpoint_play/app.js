const SIZE = 9;
const BLACK = 1;
const WHITE = 2;
const BOARD_PADDING = 62;
const PRESENTATION_MODE = document.body.dataset.mode === 'presentation';
const BOARD_CANDIDATE_LIMIT = PRESENTATION_MODE ? 3 : 10;
const LIST_CANDIDATE_LIMIT = PRESENTATION_MODE ? 3 : 12;

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const canvasFrame = document.getElementById('canvasFrame');
const candidateTooltip = document.getElementById('candidateTooltip');
const checkpoint = document.getElementById('checkpoint');
const newGameButton = document.getElementById('newGame');
const passButton = document.getElementById('pass');
const startAnalysisButton = document.getElementById('startAnalysis');
const stopAnalysisButton = document.getElementById('stopAnalysis');
const turn = document.getElementById('turn');
const turnChip = document.getElementById('turnChip');
const message = document.getElementById('message');
const analysisStatus = document.getElementById('analysisStatus');
const analysisStatusText = document.getElementById('analysisStatusText');
const rootVisits = document.getElementById('rootVisits');
const visitsPerSecond = document.getElementById('visitsPerSecond');
const rootWinrate = document.getElementById('rootWinrate');
const analysisElapsed = document.getElementById('analysisElapsed');
const candidateCount = document.getElementById('candidateCount');
const candidateList = document.getElementById('candidateList');

const integerFormatter = new Intl.NumberFormat('zh-CN');
let state = null;
let waiting = false;
let analysisCommandPending = false;
let analysis = emptyAnalysis();
let pollGeneration = 0;
let hoveredAction = null;
let hasCheckpoints = false;

function emptyAnalysis() {
  return {
    session_id: 0,
    status: 'idle',
    running: false,
    error: null,
    root_visits: 0,
    tree_size: 0,
    visits_per_second: 0,
    elapsed_seconds: 0,
    root_winrate: null,
    candidates: [],
  };
}

function formatInteger(value) {
  return integerFormatter.format(Math.max(0, Math.round(Number(value) || 0)));
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return '—';
  }
  const percent = Number(value) * 100;
  if (percent > 0 && percent < 0.1) {
    return '<0.1%';
  }
  return `${percent.toFixed(digits)}%`;
}

function formatBoardPercent(value) {
  const percent = Number(value) * 100;
  if (percent > 0 && percent < 1) {
    return '<1%';
  }
  return `${Math.round(percent)}%`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

function colorName(color) {
  return color === BLACK ? '黑方' : '白方';
}

function boardGeometry() {
  return {
    pad: BOARD_PADDING,
    step: (canvas.width - 2 * BOARD_PADDING) / (SIZE - 1),
  };
}

function boardCandidates() {
  return (analysis.candidates || [])
    .filter(candidate => !candidate.is_pass)
    .slice(0, BOARD_CANDIDATE_LIMIT);
}

function drawBoard() {
  const {pad, step} = boardGeometry();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#d2a35f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.globalAlpha = 0.09;
  ctx.strokeStyle = '#6f4b25';
  ctx.lineWidth = 1;
  for (let y = 18; y < canvas.height; y += 31) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(canvas.width * 0.32, y + 5, canvas.width * 0.68, y - 4, canvas.width, y + 2);
    ctx.stroke();
  }
  ctx.restore();

  ctx.strokeStyle = '#3f2d1d';
  ctx.lineWidth = 1.55;
  for (let index = 0; index < SIZE; index += 1) {
    const position = pad + index * step;
    ctx.beginPath();
    ctx.moveTo(pad, position);
    ctx.lineTo(canvas.width - pad, position);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(position, pad);
    ctx.lineTo(position, canvas.height - pad);
    ctx.stroke();
  }

  ctx.fillStyle = '#3d2a1b';
  for (const [row, col] of [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]]) {
    ctx.beginPath();
    ctx.arc(pad + col * step, pad + row * step, 4.6, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = 'rgba(54, 38, 24, 0.76)';
  ctx.font = '15px "Geist Mono", "SFMono-Regular", Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const columns = 'ABCDEFGHJ';
  for (let index = 0; index < SIZE; index += 1) {
    const position = pad + index * step;
    ctx.fillText(columns[index], position, 28);
    ctx.fillText(columns[index], position, canvas.height - 28);
    ctx.fillText(String(SIZE - index), 28, position);
    ctx.fillText(String(SIZE - index), canvas.width - 28, position);
  }

  if (state) {
    drawStones(pad, step);
    drawLastMove(pad, step);
  }
  drawCandidateOverlays(pad, step);
}

function drawStones(pad, step) {
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const stone = state.board[row][col];
      if (!stone) {
        continue;
      }
      const x = pad + col * step;
      const y = pad + row * step;
      ctx.save();
      ctx.shadowColor = 'rgba(36, 25, 15, 0.3)';
      ctx.shadowBlur = 7;
      ctx.shadowOffsetY = 3;
      ctx.beginPath();
      ctx.arc(x, y, step * 0.425, 0, Math.PI * 2);
      ctx.fillStyle = stone === BLACK ? '#151815' : '#f0f1eb';
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.strokeStyle = stone === BLACK ? '#30342f' : '#8b8e87';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();
    }
  }
}

function drawLastMove(pad, step) {
  if (!state.last_move) {
    return;
  }
  const [row, col] = state.last_move;
  ctx.beginPath();
  ctx.arc(pad + col * step, pad + row * step, 7, 0, Math.PI * 2);
  ctx.strokeStyle = '#b84f47';
  ctx.lineWidth = 2.8;
  ctx.stroke();
}

function drawCandidateOverlays(pad, step) {
  const candidates = boardCandidates();
  if (!candidates.length) {
    return;
  }
  const maxProbability = Math.max(candidates[0].probability || 0, 0.0001);
  for (const candidate of candidates) {
    const x = pad + candidate.col * step;
    const y = pad + candidate.row * step;
    const relative = Math.sqrt(Math.max(0, candidate.probability) / maxProbability);
    const radius = step * (0.245 + 0.075 * relative);
    const isTop = candidate.rank === 1;
    const isHovered = candidate.action === hoveredAction;

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isTop ? 'rgba(51, 121, 91, 0.96)' : 'rgba(39, 73, 59, 0.88)';
    ctx.fill();
    ctx.strokeStyle = isHovered ? '#f1f2ed' : (isTop ? '#a7d5c0' : 'rgba(224, 232, 224, 0.56)');
    ctx.lineWidth = isHovered ? 3 : 1.4;
    ctx.stroke();

    ctx.fillStyle = '#f1f2ed';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `700 ${Math.max(13, step * 0.19)}px "Geist Mono", "SFMono-Regular", Consolas, monospace`;
    ctx.fillText(formatBoardPercent(candidate.probability), x, y + 2);

    ctx.beginPath();
    ctx.arc(x - radius * 0.68, y - radius * 0.68, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#1c2a23';
    ctx.fill();
    ctx.fillStyle = '#dce9e1';
    ctx.font = '700 9px "Geist Mono", "SFMono-Regular", Consolas, monospace';
    ctx.fillText(String(candidate.rank), x - radius * 0.68, y - radius * 0.68 + 0.5);
    ctx.restore();
  }
}

function showState(next) {
  state = next;
  const humanTurn = state.to_play === state.human_color;
  turn.textContent = state.is_over
    ? state.result
    : `第 ${state.move_count + 1} 手 · ${colorName(state.to_play)}行棋`;
  turnChip.textContent = state.is_over ? '对局结束' : (humanTurn ? '轮到你' : '模型回合');
  message.className = '';
  message.textContent = state.message;
  syncControls();
  drawBoard();
}

function showError(text) {
  message.className = 'error';
  message.textContent = text;
}

function syncControls() {
  const hasGame = Boolean(state);
  const gameOver = !hasGame || state.is_over;
  const humanTurn = hasGame && state.to_play === state.human_color;
  newGameButton.disabled = waiting || analysisCommandPending || !hasCheckpoints;
  passButton.disabled = waiting || analysisCommandPending || gameOver || !humanTurn;
  startAnalysisButton.disabled = waiting || analysisCommandPending || gameOver || analysis.running;
  stopAnalysisButton.disabled = waiting || analysisCommandPending || !analysis.running;
}

function renderAnalysis() {
  const statusLabels = {
    idle: '未启动',
    starting: '初始化',
    running: '搜索中',
    stopped: '已停止',
    error: '分析错误',
  };
  const visualStatus = analysis.running ? 'running' : analysis.status;
  analysisStatus.dataset.status = visualStatus;
  analysisStatusText.textContent = statusLabels[analysis.status] || analysis.status;
  rootVisits.textContent = formatInteger(analysis.root_visits);
  visitsPerSecond.textContent = formatInteger(analysis.visits_per_second);
  rootWinrate.textContent = formatPercent(analysis.root_winrate);
  analysisElapsed.textContent = formatDuration(analysis.elapsed_seconds);
  candidateCount.textContent = `${(analysis.candidates || []).length} 个`;
  renderCandidateList();
  syncControls();
  drawBoard();
  if (analysis.error) {
    showError(`MCTS 分析失败：${analysis.error}`);
  }
}

function renderCandidateList() {
  candidateList.replaceChildren();
  const candidates = (analysis.candidates || []).slice(0, LIST_CANDIDATE_LIMIT);
  if (!candidates.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-analysis';
    const ring = document.createElement('span');
    ring.className = 'empty-ring';
    const copy = document.createElement('p');
    copy.textContent = analysis.running
      ? '正在建立根节点并收集首批候选手。'
      : '启动 MCTS 后，候选点会直接显示在棋盘上。';
    empty.append(ring, copy);
    candidateList.append(empty);
    return;
  }

  for (const candidate of candidates) {
    const row = document.createElement('div');
    row.className = 'candidate-row';
    row.dataset.action = String(candidate.action);
    if (candidate.action === hoveredAction) {
      row.classList.add('is-hovered');
    }

    const coordinateCell = document.createElement('div');
    coordinateCell.className = 'candidate-coordinate';
    const rank = document.createElement('span');
    rank.className = 'candidate-rank';
    rank.textContent = String(candidate.rank);
    const coordinateText = document.createElement('span');
    coordinateText.textContent = candidate.coordinate;
    coordinateCell.append(rank, coordinateText);

    const probabilityCell = document.createElement('div');
    probabilityCell.className = 'probability-cell';
    const probabilityBar = document.createElement('span');
    probabilityBar.className = 'probability-bar';
    probabilityBar.style.width = `${Math.max(0, Math.min(100, candidate.probability * 100))}%`;
    const probabilityText = document.createElement('strong');
    probabilityText.textContent = formatPercent(candidate.probability);
    probabilityCell.append(probabilityBar, probabilityText);

    const winrateCell = document.createElement('span');
    winrateCell.className = 'candidate-number';
    winrateCell.textContent = formatPercent(candidate.winrate);

    const visitsCell = document.createElement('span');
    visitsCell.className = 'candidate-number';
    visitsCell.textContent = formatInteger(candidate.visits);

    row.append(coordinateCell, probabilityCell, winrateCell, visitsCell);
    row.addEventListener('mouseenter', () => {
      hoveredAction = candidate.action;
      renderCandidateHoverState();
    });
    row.addEventListener('mouseleave', () => {
      hoveredAction = null;
      renderCandidateHoverState();
    });
    candidateList.append(row);
  }
}

function renderCandidateHoverState() {
  for (const row of candidateList.querySelectorAll('.candidate-row')) {
    row.classList.toggle('is-hovered', Number(row.dataset.action) === hoveredAction);
  }
  drawBoard();
}

async function requestJSON(path, options = {}) {
  const response = await fetch(path, {
    cache: 'no-store',
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '请求失败');
  }
  return data;
}

async function post(path, payload = {}) {
  return requestJSON(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
}

function cancelAnalysisPolling() {
  pollGeneration += 1;
}

function beginAnalysisPolling() {
  const generation = ++pollGeneration;
  const poll = async () => {
    if (generation !== pollGeneration) {
      return;
    }
    try {
      const next = await requestJSON('/api/analysis');
      if (generation !== pollGeneration) {
        return;
      }
      analysis = next;
      renderAnalysis();
      if (analysis.running) {
        window.setTimeout(poll, 400);
      }
    } catch (error) {
      if (generation === pollGeneration) {
        showError(error.message);
        window.setTimeout(poll, 1000);
      }
    }
  };
  window.setTimeout(poll, 220);
}

async function startAnalysis() {
  if (analysis.running || analysisCommandPending || waiting || !state || state.is_over) {
    return;
  }
  analysisCommandPending = true;
  analysis = {...analysis, status: 'starting', running: true, error: null};
  renderAnalysis();
  try {
    analysis = await post('/api/analysis/start');
    renderAnalysis();
    beginAnalysisPolling();
  } catch (error) {
    analysis = {...emptyAnalysis(), status: 'error', error: error.message};
    renderAnalysis();
  } finally {
    analysisCommandPending = false;
    syncControls();
  }
}

async function stopAnalysis() {
  if (!analysis.running || analysisCommandPending) {
    return;
  }
  cancelAnalysisPolling();
  analysisCommandPending = true;
  analysisStatusText.textContent = '正在停止';
  syncControls();
  try {
    analysis = await post('/api/analysis/stop');
    renderAnalysis();
  } catch (error) {
    showError(error.message);
  } finally {
    analysisCommandPending = false;
    syncControls();
  }
}

function clearAnalysisForPositionChange() {
  cancelAnalysisPolling();
  hoveredAction = null;
  candidateTooltip.hidden = true;
  analysis = emptyAnalysis();
  renderAnalysis();
}

async function withGameWaiting(action) {
  if (waiting || analysisCommandPending) {
    return;
  }
  waiting = true;
  const wasAnalyzing = analysis.running;
  clearAnalysisForPositionChange();
  message.className = '';
  message.textContent = wasAnalyzing
    ? '正在停止分析并计算模型应手…'
    : '模型思考中…';
  syncControls();
  try {
    showState(await action());
  } catch (error) {
    showError(error.message);
  } finally {
    waiting = false;
    syncControls();
  }
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    rect,
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  };
}

function candidateAtPoint(x, y) {
  const {pad, step} = boardGeometry();
  for (const candidate of boardCandidates()) {
    const dx = x - (pad + candidate.col * step);
    const dy = y - (pad + candidate.row * step);
    if (Math.hypot(dx, dy) <= step * 0.39) {
      return candidate;
    }
  }
  return null;
}

canvas.addEventListener('pointermove', event => {
  const point = canvasPoint(event);
  const candidate = candidateAtPoint(point.x, point.y);
  hoveredAction = candidate ? candidate.action : null;
  renderCandidateHoverState();
  if (!candidate) {
    candidateTooltip.hidden = true;
    return;
  }
  const frameRect = canvasFrame.getBoundingClientRect();
  candidateTooltip.textContent = [
    `${candidate.coordinate} · 候选 ${candidate.rank}`,
    `搜索占比 ${formatPercent(candidate.probability)}  胜率 ${formatPercent(candidate.winrate)}`,
    `${formatInteger(candidate.visits)} visits  策略先验 ${formatPercent(candidate.prior)}`,
  ].join('\n');
  candidateTooltip.style.left = `${event.clientX - frameRect.left + 14}px`;
  candidateTooltip.style.top = `${event.clientY - frameRect.top + 14}px`;
  candidateTooltip.hidden = false;
});

canvas.addEventListener('pointerleave', () => {
  hoveredAction = null;
  candidateTooltip.hidden = true;
  renderCandidateHoverState();
});

canvas.addEventListener('click', event => {
  if (!state || state.is_over || waiting || analysisCommandPending || state.to_play !== state.human_color) {
    return;
  }
  const {x, y} = canvasPoint(event);
  const {pad, step} = boardGeometry();
  const col = Math.round((x - pad) / step);
  const row = Math.round((y - pad) / step);
  if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) {
    return;
  }
  const intersectionX = pad + col * step;
  const intersectionY = pad + row * step;
  if (Math.hypot(x - intersectionX, y - intersectionY) > step * 0.45) {
    return;
  }
  withGameWaiting(() => post('/api/move', {row, col}));
});

newGameButton.addEventListener('click', () => withGameWaiting(() => post('/api/new-game', {
  checkpoint: checkpoint.value,
  color: document.querySelector('input[name="color"]:checked').value,
})));

passButton.addEventListener('click', () => withGameWaiting(() => post('/api/pass')));
startAnalysisButton.addEventListener('click', startAnalysis);
stopAnalysisButton.addEventListener('click', stopAnalysis);

document.addEventListener('keydown', event => {
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) {
    return;
  }
  if (event.code === 'Space') {
    event.preventDefault();
    if (analysis.running) {
      stopAnalysis();
    } else {
      startAnalysis();
    }
  } else if (event.key === 'Escape' && analysis.running) {
    event.preventDefault();
    stopAnalysis();
  }
});

async function initialize() {
  drawBoard();
  renderAnalysis();
  try {
    const [checkpointData, stateData, analysisData] = await Promise.all([
      requestJSON('/api/checkpoints'),
      requestJSON('/api/state'),
      requestJSON('/api/analysis'),
    ]);

    checkpoint.replaceChildren();
    for (const name of checkpointData.checkpoints) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      checkpoint.append(option);
    }
    hasCheckpoints = checkpointData.checkpoints.length > 0;

    if (stateData.state) {
      showState(stateData.state);
    }
    analysis = analysisData;
    renderAnalysis();
    if (analysis.running) {
      beginAnalysisPolling();
    }
    syncControls();
  } catch (error) {
    showError(error.message);
  }
}

initialize();
