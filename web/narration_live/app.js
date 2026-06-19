const connectionDot = document.getElementById("connectionDot");
const connectionText = document.getElementById("connectionText");
const pipelineButton = document.getElementById("pipelineButton");
const audioButton = document.getElementById("audioButton");
const demoButton = document.getElementById("demoButton");
const resetButton = document.getElementById("resetButton");
const queueCount = document.getElementById("queueCount");
const audioPlayer = document.getElementById("audioPlayer");
const playerStatus = document.getElementById("playerStatus");
const playerHint = document.getElementById("playerHint");
const liveType = document.getElementById("liveType");
const liveText = document.getElementById("liveText");
const actionCount = document.getElementById("actionCount");
const eventList = document.getElementById("eventList");
const videoFeed = document.getElementById("videoFeed");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const pipelineStatus = document.getElementById("pipelineStatus");
const scoreA = document.getElementById("scoreA");
const scoreB = document.getElementById("scoreB");
const summaryOverlay = document.getElementById("summaryOverlay");
const summaryClose = document.getElementById("summaryClose");
const summaryScoreA = document.getElementById("summaryScoreA");
const summaryScoreB = document.getElementById("summaryScoreB");
const summaryResult = document.getElementById("summaryResult");
const statGoalsA = document.getElementById("statGoalsA");
const statPassesA = document.getElementById("statPassesA");
const statShotsA = document.getElementById("statShotsA");
const statControlsA = document.getElementById("statControlsA");
const statOffsidesA = document.getElementById("statOffsidesA");
const statGoalsB = document.getElementById("statGoalsB");
const statPassesB = document.getElementById("statPassesB");
const statShotsB = document.getElementById("statShotsB");
const statControlsB = document.getElementById("statControlsB");
const statOffsidesB = document.getElementById("statOffsidesB");
const summaryActions = document.getElementById("summaryActions");

const audioQueue = [];
const rows = new Map();
const queuedAudioUrls = new Set();
let audioEnabled = false;
let isPlaying = false;
let lastAudioUrl = null;
let lastSpokenText = null;
let totalActions = 0;
let pipelineRunning = false;

const source = new EventSource("/events");

source.addEventListener("connected", () => {
  setConnection("Conectado", "connected");
});

source.addEventListener("action_received", (event) => {
  const payload = JSON.parse(event.data);
  applyRecord(payload);
});

source.addEventListener("narration_started", (event) => {
  const payload = JSON.parse(event.data);
  updateRowState(payload.id, "Narrando");
});

source.addEventListener("narration_ready", (event) => {
  const payload = JSON.parse(event.data);
  applyRecord(payload);
});

source.addEventListener("narration_error", (event) => {
  const payload = JSON.parse(event.data);
  updateErrorRow(payload.id, payload.error);
});

source.addEventListener("demo_started", () => {
  demoButton.disabled = true;
});

source.addEventListener("demo_finished", () => {
  demoButton.disabled = false;
});

source.addEventListener("reset", () => {
  resetLocalPlayback();
  eventList.replaceChildren();
  rows.clear();
  queuedAudioUrls.clear();
  totalActions = 0;
  actionCount.textContent = "0";
  liveType.textContent = "Esperando";
  liveText.textContent = "Sin acciones";
});

source.addEventListener("pipeline_status", (event) => {
  const payload = JSON.parse(event.data);
  if (payload.status === "running") {
    pipelineRunning = true;
    pipelineButton.disabled = true;
    pipelineStatus.textContent = payload.step || "Procesando";
    pipelineStatus.className = "pipeline-status running";
    videoFeed.src = "/video-feed";
    videoFeed.style.display = "block";
    videoPlaceholder.style.display = "none";
  } else if (payload.status === "finished") {
    pipelineRunning = false;
    pipelineButton.disabled = false;
    pipelineStatus.textContent = "Terminado";
    pipelineStatus.className = "pipeline-status finished";
    if (payload.score) {
      scoreA.textContent = payload.score.robot_a || 0;
      scoreB.textContent = payload.score.robot_b || 0;
    }
  }
});

source.addEventListener("narration_complete", (event) => {
  const payload = JSON.parse(event.data);
  showSummary(payload.score);
});

source.addEventListener("pipeline_progress", (event) => {
  const payload = JSON.parse(event.data);
  const pct = Math.round((payload.frame / payload.total) * 100);
  pipelineStatus.textContent = `Frame ${payload.frame}/${payload.total} (${pct}%)`;
  if (payload.score) {
    scoreA.textContent = payload.score.robot_a || 0;
    scoreB.textContent = payload.score.robot_b || 0;
  }
});

source.onerror = () => {
  setConnection("Reconectando", "error");
};

setInterval(syncHistory, 1500);

pipelineButton.addEventListener("click", async () => {
  pipelineButton.disabled = true;
  pipelineStatus.textContent = "Iniciando...";
  pipelineStatus.className = "pipeline-status running";
  await fetch("/api/pipeline", { method: "POST" });
});

audioButton.addEventListener("click", async () => {
  audioEnabled = true;
  await unlockAudioOutput();
  audioButton.classList.add("primary");
  setAudioButton("Audio activo", "✓");
  playerStatus.textContent = "Audio activo";
  playerHint.textContent = "Reproduciendo bienvenida.";
  if (!isPlaying && audioQueue.length === 0) {
    audioQueue.push("/assets/bienvenida.mpeg");
    queueCount.textContent = String(audioQueue.length);
  }
  playNext();
});

demoButton.addEventListener("click", async () => {
  demoButton.disabled = true;
  await fetch("/api/demo", { method: "POST" });
});

resetButton.addEventListener("click", async () => {
  resetLocalPlayback();
  await fetch("/api/reset", { method: "POST" });
});

audioPlayer.addEventListener("play", () => {
  isPlaying = true;
  setAudioButton("Reproduciendo", "▶");
  playerStatus.textContent = "Reproduciendo";
});

audioPlayer.addEventListener("ended", () => {
  isPlaying = false;
  setAudioButton("Audio activo", "✓");
  playerStatus.textContent = "Audio activo";
  playNext();
});

audioPlayer.addEventListener("error", () => {
  isPlaying = false;
  setAudioButton("Reintentar audio", "▶");
  playerStatus.textContent = "No se pudo reproducir";
  playerHint.textContent = "Presiona Audio o usa el control manual para reintentar.";
});

function setConnection(text, state) {
  connectionText.textContent = text;
  connectionDot.className = `status-dot ${state}`;
}

function resetLocalPlayback() {
  audioQueue.length = 0;
  queuedAudioUrls.clear();
  queueCount.textContent = "0";
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  lastAudioUrl = null;
  lastSpokenText = null;
  isPlaying = false;
  if (audioEnabled) {
    setAudioButton("Audio activo", "✓");
    playerStatus.textContent = "Audio activo";
    playerHint.textContent = "Esperando narracion.";
  } else {
    setAudioButton("Audio", "▶");
    playerStatus.textContent = "Audio listo";
    playerHint.textContent = "Activa Audio antes de iniciar la demo.";
  }
}

function addActionRow(payload) {
  if (rows.has(payload.id)) return;
  const action = payload.action;
  const row = document.createElement("article");
  row.className = `event-row ${teamClass(action)}`;
  row.dataset.id = payload.id;

  const time = document.createElement("div");
  time.className = "event-time";
  time.textContent = action.timestamp || `#${payload.id}`;

  const main = document.createElement("div");
  main.className = "event-main";

  const title = document.createElement("strong");
  title.textContent = actionTitle(action);

  const detail = document.createElement("span");
  detail.textContent = actionDetail(action);

  main.append(title, detail);

  const state = document.createElement("div");
  state.className = "event-state";
  state.textContent = "Cola";

  row.append(time, main, state);
  eventList.prepend(row);
  rows.set(payload.id, { row, title, detail, state, action });
  totalActions = rows.size;
  actionCount.textContent = String(totalActions);

  liveType.textContent = action.type || "Accion";
  liveText.textContent = actionDetail(action);
}

function applyRecord(record) {
  if (record.action) {
    addActionRow(record);
  }
  if (record.status === "narrating") {
    updateRowState(record.id, "Narrando");
  } else if (record.status === "ready" || record.text) {
    updateReadyRow(record);
    enqueueAudio(record);
  } else if (record.status === "error" || record.error) {
    updateErrorRow(record.id, record.error || "Error de narracion");
  }
}

async function syncHistory() {
  try {
    const response = await fetch("/api/history", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    for (const record of payload.records || []) {
      applyRecord(record);
    }
  } catch {
    setConnection("Reconectando", "error");
  }
}

function enqueueAudio(record) {
  if (!record.audio_url) {
    if (record.text) {
      lastSpokenText = record.text;
      playerStatus.textContent = "Narracion con voz del navegador";
      playerHint.textContent = "ElevenLabs no entrego audio; usando fallback local.";
      if (audioEnabled) {
        speakWithBrowser(record.text);
      }
    }
    return;
  }
  if (queuedAudioUrls.has(record.audio_url)) return;
  queuedAudioUrls.add(record.audio_url);
  audioQueue.push(record.audio_url);
  lastAudioUrl = record.audio_url;
  queueCount.textContent = String(audioQueue.length);
  playerStatus.textContent = "Audio recibido";
  playerHint.textContent = "Reproduciendo automaticamente o usa el control manual.";
  playNext();
}

function speakWithBrowser(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "es-MX";
  utterance.rate = 1.18;
  utterance.pitch = 1;
  utterance.onstart = () => {
    isPlaying = true;
    setAudioButton("Reproduciendo", "▶");
    playerStatus.textContent = "Reproduciendo fallback";
  };
  utterance.onend = () => {
    isPlaying = false;
    setAudioButton("Audio activo", "✓");
    playerStatus.textContent = "Audio activo";
    playNext();
  };
  utterance.onerror = () => {
    isPlaying = false;
    setAudioButton("Reintentar audio", "▶");
    playerStatus.textContent = "No se pudo reproducir fallback";
  };
  window.speechSynthesis.speak(utterance);
}

function updateRowState(id, text) {
  const entry = rows.get(id);
  if (!entry) return;
  entry.state.className = "event-state";
  entry.state.textContent = text;
}

function updateReadyRow(payload) {
  const entry = rows.get(payload.id);
  if (!entry) return;
  entry.detail.textContent = payload.text;
  entry.state.className = "event-state ready";
  entry.state.textContent = "Audio";
  liveType.textContent = payload.action.type || "Narracion";
  liveText.textContent = payload.text;
}

function updateErrorRow(id, error) {
  const entry = rows.get(id);
  if (!entry) return;
  entry.detail.textContent = error;
  entry.state.className = "event-state error";
  entry.state.textContent = "Error";
}

function playNext() {
  if (!audioEnabled || isPlaying || audioQueue.length === 0) return;
  const url = audioQueue.shift();
  lastAudioUrl = url;
  queueCount.textContent = String(audioQueue.length);
  isPlaying = true;

  audioPlayer.pause();
  audioPlayer.src = url;
  audioPlayer.load();
  setAudioButton("Reproduciendo", "▶");
  playerStatus.textContent = "Reproduciendo";
  playerHint.textContent = url;

  audioPlayer.play().catch(() => {
    audioQueue.unshift(url);
    queueCount.textContent = String(audioQueue.length);
    isPlaying = false;
    audioPlayer.src = url;
    setAudioButton("Reintentar audio", "▶");
    playerStatus.textContent = "Reproduccion bloqueada";
    playerHint.textContent = "Usa el control de audio o presiona Audio otra vez.";
  });
}

function setAudioButton(label, icon) {
  audioButton.replaceChildren();
  const iconNode = document.createElement("span");
  iconNode.className = "icon";
  iconNode.textContent = icon;
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  audioButton.append(iconNode, labelNode);
}

async function unlockAudioOutput() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  const context = new AudioContextClass();
  if (context.state === "suspended") {
    await context.resume();
  }
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.frequency.value = 880;
  gain.gain.value = 0.025;
  oscillator.connect(gain);
  gain.connect(context.destination);
  oscillator.start();
  oscillator.stop(context.currentTime + 0.08);
}

function actionTitle(action) {
  const type = action.type || action.action || action.accion || "accion";
  const team = action.team || action.equipo;
  const robot = action.robot_id || action.robot || action.player;
  return [type, team, robot].filter(Boolean).join(" · ");
}

function teamClass(action) {
  const team = String(action.team || action.equipo || "").toLowerCase();
  if (team === "blanco") return "team-blanco";
  if (team === "negro") return "team-negro";
  return "";
}

function actionDetail(action) {
  const parts = [];
  if (action.target_robot_id || action.target || action.receiver) {
    parts.push(`destino ${action.target_robot_id || action.target || action.receiver}`);
  }
  if (action.outcome || action.resultado) {
    parts.push(action.outcome || action.resultado);
  }
  if (typeof action.confidence === "number") {
    parts.push(`${Math.round(action.confidence * 100)}%`);
  }
  if (action.score || action.marcador) {
    const score = action.score || action.marcador;
    parts.push(Object.entries(score).map(([team, goals]) => `${team} ${goals}`).join(" - "));
  }
  return parts.join(" · ") || "Procesando accion";
}

function showSummary(score) {
  const a = score ? (score.robot_a || 0) : 0;
  const b = score ? (score.robot_b || 0) : 0;
  summaryScoreA.textContent = a;
  summaryScoreB.textContent = b;

  if (a > b) summaryResult.textContent = "Robot A gana!";
  else if (b > a) summaryResult.textContent = "Robot B gana!";
  else summaryResult.textContent = "Empate!";

  const stats = { a: { gol: 0, pase: 0, tiro: 0, controla: 0, fuera_de_lugar: 0 },
                  b: { gol: 0, pase: 0, tiro: 0, controla: 0, fuera_de_lugar: 0 } };

  rows.forEach((entry) => {
    if (!entry.action) return;
    const team = String(entry.action.team || entry.action.equipo || "").toLowerCase();
    const kind = String(entry.action.type || entry.action.action || entry.action.accion || "").toLowerCase();
    const side = team.includes("robot_a") || team.includes("blanco") || team === "robot_a" ? "a" : "b";
    if (kind === "gol" || kind === "goal") stats[side].gol++;
    else if (kind === "pase" || kind === "pass") stats[side].pase++;
    else if (kind === "tiro" || kind === "shot") stats[side].tiro++;
    else if (kind === "controla" || kind === "control") stats[side].controla++;
    else if (kind === "fuera_de_lugar" || kind === "offside") stats[side].fuera_de_lugar++;
  });

  statGoalsA.textContent = stats.a.gol;
  statPassesA.textContent = stats.a.pase;
  statShotsA.textContent = stats.a.tiro;
  statControlsA.textContent = stats.a.controla;
  statOffsidesA.textContent = stats.a.fuera_de_lugar;
  statGoalsB.textContent = stats.b.gol;
  statPassesB.textContent = stats.b.pase;
  statShotsB.textContent = stats.b.tiro;
  statControlsB.textContent = stats.b.controla;
  statOffsidesB.textContent = stats.b.fuera_de_lugar;

  summaryActions.replaceChildren();
  rows.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "popup-action-item";
    const label = document.createElement("strong");
    label.textContent = entry.title.textContent;
    const status = document.createElement("span");
    status.className = "popup-action-status";
    const stateText = entry.state.textContent;
    status.textContent = stateText;
    if (stateText.toLowerCase() === "error") status.classList.add("error");
    item.append(label, status);
    summaryActions.appendChild(item);
  });

  summaryOverlay.classList.add("visible");
}

function hideSummary() {
  summaryOverlay.classList.remove("visible");
}

summaryClose.addEventListener("click", hideSummary);
summaryOverlay.addEventListener("click", (e) => {
  if (e.target === summaryOverlay) hideSummary();
});

