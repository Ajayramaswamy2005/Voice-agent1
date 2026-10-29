const $ = (id) => document.getElementById(id);
const STAGES = [
  ["stt", "STT final", "--s-stt"],
  ["llm_ttft", "LLM TTFT", "--s-llm"],
  ["chunking", "phrase", "--s-chunk"],
  ["tts_ttfb", "TTS TTFB", "--s-tts"],
  ["turn_gate", "turn gate", "--s-gate"],
  ["playout", "playout", "--s-play"],
];
const OUT_RATE = 24000;

let ws, ctx, micNode, player, stream, analyser, outAnalyser;
let dropBelow = 0;
const firstRecv = new Map(); // seq -> performance.now() of first frame
let interimEl = null;
let lines = [];
let line = null; // selected line from /api/personas
let code = "";
try { code = sessionStorage.getItem("code") || ""; } catch {}

const STATE_TEXT = { idle: "Ready to call", listening: "Listening", thinking: "Thinking", speaking: "Speaking" };
function setState(s) {
  $("orb").dataset.state = s;
  $("state").textContent = STATE_TEXT[s] || s;
}

function add(kind, text, extra = "") {
  $("empty")?.remove();
  const el = document.createElement("div");
  if (kind === "event") {
    el.className = `event ${extra}`;
    el.textContent = text;
  } else {
    el.className = `msg ${kind} ${extra}`;
    el.append(document.createTextNode(text));
  }
  $("feed").append(el);
  $("feed").scrollTop = $("feed").scrollHeight;
  return el;
}

function showLatency(r) {
  $("latTotal").innerHTML = `${Math.round(r.e2e)}<small>ms${r.path === "filler" ? ", filler clip" : ""}</small>`;
  const bar = $("latBar");
  bar.querySelectorAll("span").forEach((s) => s.remove());
  const legend = $("latLegend");
  legend.innerHTML = "";
  for (const [k, label, color] of STAGES) {
    const seg = document.createElement("span");
    seg.style.width = `${Math.min(100, (r[k] / 1000) * 100)}%`;
    seg.style.background = `var(${color})`;
    seg.title = `${label}: ${r[k]} ms`;
    bar.append(seg); // the 700 ms marker is absolutely positioned, so order is unaffected
    legend.insertAdjacentHTML("beforeend", `<div><i style="background:var(${color})"></i>${label}<b>${Math.round(r[k])}</b></div>`);
  }
}

function onBinary(buf) {
  const seq = new DataView(buf).getUint32(0, true);
  if (seq !== 0 && seq < dropBelow) return;
  const pcm = new Int16Array(buf, 4);
  const ratio = OUT_RATE / ctx.sampleRate;
  const n = Math.floor(pcm.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * ratio, j = Math.floor(p), f = p - j;
    const a = pcm[j] / 32768, b = (pcm[j + 1] ?? pcm[j]) / 32768;
    out[i] = a + (b - a) * f;
  }
  if (seq !== 0 && !firstRecv.has(seq)) firstRecv.set(seq, performance.now());
  player.port.postMessage({ type: "audio", seq, side: seq === 0, data: out }, [out.buffer]);
}

function onMessage(m) {
  switch (m.type) {
    case "user_text":
      if (!interimEl) interimEl = add("user", "", "interim");
      interimEl.lastChild.textContent = (interimEl.dataset.final || "") + m.text;
      if (m.final) interimEl.dataset.final = (interimEl.dataset.final || "") + m.text + " ";
      setState("listening");
      break;
    case "turn":
      add("event", m.complete ? "You finished your turn" : "Waiting: you sound mid-thought");
      if (m.complete) {
        if (interimEl) interimEl.classList.remove("interim");
        interimEl = null;
        setState("thinking");
      }
      break;
    case "agent_text":
      if (m.text) add("agent", m.text);
      break;
    case "tool":
      add("event", `Using ${m.name.replaceAll("_", " ")}`, "tool");
      break;
    case "backchannel":
      add("event", `Said “${m.text}” while you talked`);
      break;
    case "duck":
      player.port.postMessage({ type: "gain", value: m.gain ?? 0.6 });
      break;
    case "unduck":
      player.port.postMessage({ type: "gain", value: 1 });
      break;
    case "clear":
      dropBelow = m.drop_below;
      player.port.postMessage({ type: "clear" });
      player.port.postMessage({ type: "gain", value: 1 });
      break;
    case "barge_in": {
      add("event", `You cut in after ${(m.played_ms / 1000).toFixed(1)} s`, "barge");
      if (m.heard) add("agent", m.heard + " —", "cut");
      if (interimEl) interimEl.classList.remove("interim");
      interimEl = null;
      setState("listening");
      break;
    }
    case "latency":
      showLatency(m);
      break;
    case "app_event":
      seenEvents.add(m.text);
      add("event", m.text, "app");
      refreshBrowser();
      break;
    case "ended":
      add("event", m.reason);
      break;
  }
}

async function startCall() {
  ctx = new AudioContext({ latencyHint: "interactive" });
  await ctx.audioWorklet.addModule("/static/worklets.js");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const src = ctx.createMediaStreamSource(stream);
  micNode = new AudioWorkletNode(ctx, "mic-processor");
  analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  src.connect(micNode);
  src.connect(analyser);
  player = new AudioWorkletNode(ctx, "player-processor", { outputChannelCount: [2] });
  player.connect(ctx.destination);
  outAnalyser = ctx.createAnalyser(); // the orb also moves with the agent's voice, not only yours
  outAnalyser.fftSize = 512;
  player.connect(outAnalyser);
  player.port.onmessage = (e) => {
    const m = e.data;
    if (m.type === "started") {
      setState("speaking");
      const t = firstRecv.get(m.seq);
      if (t !== undefined && ws?.readyState === 1) {
        const outLat = (ctx.outputLatency || ctx.baseLatency || 0) * 1000;
        ws.send(JSON.stringify({ type: "playout", seq: m.seq, delay_ms: performance.now() - t + outLat }));
      }
    } else if (m.type === "idle") setState("listening");
  };

  const params = new URLSearchParams({ persona: line.key, lang: $("lang").value, code });
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?${params}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => (typeof e.data === "string" ? onMessage(JSON.parse(e.data)) : onBinary(e.data));
  ws.onclose = (e) => {
    if (e.code === 4401) askCode();
    else if (e.reason) add("event", e.reason, "barge");
    stopCall();
  };
  micNode.port.onmessage = (e) => ws.readyState === 1 && ws.send(e.data);

  const lvl = new Uint8Array(analyser.fftSize);
  (function meter() {
    if (!analyser) return;
    let peak = 0;
    for (const a of [analyser, outAnalyser]) {
      if (!a) continue;
      a.getByteTimeDomainData(lvl);
      for (const v of lvl) peak = Math.max(peak, Math.abs(v - 128));
    }
    $("orb").style.setProperty("--lvl", Math.min(1, peak / 50));
    requestAnimationFrame(meter);
  })();

  $("say").disabled = $("sayBtn").disabled = false;
  $("say").placeholder = "Type to the agent instead of speaking";
  $("callLabel").textContent = "Hang up";
  $("call").classList.add("live");
  document.querySelectorAll(".langs button").forEach((b) => (b.disabled = true));
  $("lines").querySelectorAll("button").forEach((b) => (b.disabled = true));
  setState("listening");
}

function stopCall() {
  ws?.readyState === 1 && ws.close();
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  ws = ctx = stream = analyser = outAnalyser = null;
  $("orb").style.setProperty("--lvl", 0);
  firstRecv.clear();
  dropBelow = 0;
  $("say").disabled = $("sayBtn").disabled = true;
  $("say").placeholder = "Start a call to type to the agent";
  $("callLabel").textContent = "Call";
  $("call").classList.remove("live");
  document.querySelectorAll(".langs button").forEach((b) => (b.disabled = false));
  $("lines").querySelectorAll("button").forEach((b) => (b.disabled = false));
  setState("idle");
}

$("call").onclick = () => (ws ? stopCall() : startCall().catch((e) => { alert(e.message); stopCall(); }));

// -- lines -----------------------------------------------------------------------------------------
const TRY = {
  clinic: "Try: “I need a skin doctor tomorrow morning”, then cut in with “wait, Friday instead”.",
  support: "Try: “मेरा ऑर्डर OD1005 अभी तक नहीं आया”, then “can you make it high priority?”",
  browser: "Try: “Fill this form with my resume details.” It asks before submitting anything.",
};

function pickLine(key) {
  line = lines.find((l) => l.key === key) || lines[0];
  for (const b of $("lines").children) b.setAttribute("aria-pressed", String(b.dataset.key === line.key));
  $("lineName").textContent = line.name;
  $("tools").innerHTML = "";
  for (const t of line.tools) {
    const chip = document.createElement("span");
    chip.textContent = t.replaceAll("_", " ");
    $("tools").append(chip);
  }
  if ($("tryThis")) $("tryThis").textContent = TRY[line.key] || "Interrupt it mid-sentence or pause mid-thought.";
  $("browser").hidden = !line.browser;
  $("main").classList.toggle("with-browser", line.browser);
  if (line.browser) { refreshBrowser(); loadResume(); }
  try { localStorage.setItem("line", line.key); } catch {}
}

fetch("/api/personas").then((r) => r.json()).then((ps) => {
  lines = ps;
  for (const l of ps) {
    const b = document.createElement("button");
    b.dataset.key = l.key;
    b.textContent = l.name.split(":")[0];
    b.title = l.name;
    b.onclick = () => pickLine(l.key);
    $("lines").append(b);
  }
  let saved = new URLSearchParams(location.search).get("line"); // links from the landing page
  if (!saved) try { saved = localStorage.getItem("line"); } catch {}
  pickLine(saved);
});

// -- access code ------------------------------------------------------------------------------------
function askCode() {
  $("codeInput").value = code;
  $("codeDialog").showModal();
}
$("codeForm").onsubmit = () => {
  code = $("codeInput").value.trim();
  try { sessionStorage.setItem("code", code); } catch {}
  if (line?.browser) { refreshBrowser(); loadResume(); }
};
fetch("/api/config").then((r) => r.json()).then((c) => { if (c.access_code_required && !code) askCode(); });

// -- browser panel ----------------------------------------------------------------------------------
const B_STATE = {
  idle: "Idle", planning: "Planning", plan_review: "Plan ready", running: "Working",
  needs_approval: "Needs your OK", waiting_quota: "Waiting for quota", done: "Done", failed: "Stopped", stopped: "Stopped",
};
let screenVersion = -1;
let pollTimer = null;
let lastEventCount = null;
let editingPlan = false;
let lastStatus = null;
let planSig = "", stepsSig = "";
let doneBefore = new Set(); // plan steps already ticked, so only a newly finished one animates
const seenEvents = new Set(); // already shown in the conversation via the call

function api(path, opts = {}) {
  return fetch(path, { ...opts, headers: { ...(opts.headers || {}), "x-access-code": code } });
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

function describe(a) {
  const target = a.mark ? `#${a.mark}` : a.x != null ? `at ${a.x}, ${a.y}` : "";
  const text = a.text ? ` “${String(a.text).slice(0, 40)}”` : "";
  return `${a.action.replaceAll("_", " ")} ${target}${text}`.trim();
}

// Updates that arrive while there's no call (typed tasks) go into the conversation and, if this tab is
// in the background, a desktop notification.
function onNewEvents(st) {
  if (lastEventCount === null) { lastEventCount = st.event_count; return; }
  const fresh = st.events.slice(Math.max(0, st.events.length - (st.event_count - lastEventCount)));
  lastEventCount = st.event_count;
  for (const text of fresh) {
    if (seenEvents.has(text)) continue;
    if (!ws) add("event", text, "app");
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
      try { new Notification("Kiran, browser assistant", { body: text }); } catch {}
    }
  }
}

function renderPlan(st) {
  const review = st.state === "plan_review";
  $("plan").hidden = !st.goal;
  $("bGoal").textContent = st.goal;
  $("planHint").textContent = review ? "Waiting for your OK" : st.plan.every((p) => p.done) && st.plan.length ? "All done" : "";
  $("planActions").hidden = !review;
  if (!review) { editingPlan = false; $("planText").hidden = true; }
  if (editingPlan) return; // don't overwrite what the user is typing
  const working = ["running", "needs_approval", "waiting_quota"].includes(st.state);
  const current = st.plan.findIndex((p) => !p.done);
  // Rebuild only when something changed: rebuilding restarts the CSS animations on every poll.
  const sig = JSON.stringify([st.goal, st.plan, working, st.state === "done"]);
  if (sig === planSig) return;
  if (st.goal !== JSON.parse(planSig || '[""]')[0]) doneBefore = new Set();
  planSig = sig;
  const list = $("planList");
  list.innerHTML = "";
  st.plan.forEach((p, i) => {
    const li = document.createElement("li");
    li.textContent = p.text;
    if (p.done || st.state === "done") {
      li.className = doneBefore.has(i) ? "done settled" : "done";
      doneBefore.add(i);
    } else if (working && i === current) li.className = "now";
    list.append(li);
  });
  $("planEditBtn").textContent = "Edit plan";
}

async function refreshBrowser() {
  clearTimeout(pollTimer);
  if (!line?.browser) return;
  let st = null;
  try {
    const r = await api("/api/browser");
    if (r.ok) st = await r.json();
  } catch {}
  if (!st) { pollTimer = setTimeout(refreshBrowser, 5000); return; }
  lastStatus = st;
  onNewEvents(st);
  const busy = ["planning", "plan_review", "running", "needs_approval", "waiting_quota"].includes(st.state);
  $("bState").dataset.state = st.state;
  $("bState").textContent = B_STATE[st.state] || st.state;
  $("bStop").hidden = !busy;
  $("frame").classList.toggle("busy", st.state === "running" || st.state === "planning");
  $("bWindow").textContent = st.window_visible ? "Hide window" : "Show window";
  $("bWindow").setAttribute("aria-pressed", String(st.window_visible));
  if (document.activeElement !== $("url")) $("url").value = st.url || "";
  $("bQueue").textContent = st.queue.length ? `Next up: ${st.queue.join("; ")}` : "";
  renderPlan(st);

  const action = st.waiting_for_approval && st.approval_kind === "action";
  $("approve").hidden = !action;
  $("approveText").textContent = action ? `About to ${st.waiting_for_approval}. Go ahead?` : "";

  const finished = ["done", "failed", "stopped"].includes(st.state);
  $("result").hidden = !(finished && (st.summary || st.answer));
  $("bSummary").textContent = st.summary || "";
  $("bAnswer").textContent = st.answer || "";

  const steps = $("bSteps");
  const sSig = JSON.stringify(st.steps);
  if (sSig !== stepsSig) {
  const seen = new Set([...steps.children].map((li) => li.value));
  stepsSig = sSig;
  steps.innerHTML = "";
  for (const s of st.steps) {
    const li = document.createElement("li");
    li.value = s.n;
    li.textContent = s.observation;
    if (s.saw_image) {
      const eye = document.createElement("span");
      eye.className = "img";
      eye.textContent = "with screenshot";
      li.append(eye);
    }
    const acts = document.createElement("div");
    acts.className = "acts";
    acts.textContent = s.actions
      .map((a, i) => describe(a) + (s.results[i] && s.results[i] !== "ok" ? `: ${s.results[i]}` : ""))
      .join("; ");
    if (s.results.some((r) => r.startsWith("failed") || r.startsWith("blocked") || r.startsWith("model:"))) acts.classList.add("bad");
    li.append(acts);
    if (seen.has(s.n)) li.classList.add("settled"); // only new steps animate in
    steps.append(li);
  }
  steps.scrollTop = steps.scrollHeight;
  }
  $("bSites").textContent = st.allowed_sites.length ? `Allowed sites: ${st.allowed_sites.join(", ")}` : "Allowed sites: any";
  $("bTabs").textContent = st.tabs > 1 ? `${st.tabs} tabs` : "";
  if (st.screen_version !== screenVersion) {
    screenVersion = st.screen_version;
    const img = new Image();
    img.alt = `What the agent sees on ${st.url || "the page"}`;
    img.onload = () => $("screen").replaceChildren(img);
    img.src = `/api/browser/screen.jpg?v=${screenVersion}&code=${encodeURIComponent(code)}`;
  }
  if (!$("audit").hidden) loadAudit();
  // Poll fast while the agent works, slowly otherwise.
  pollTimer = setTimeout(refreshBrowser, busy ? 800 : 4000);
}

async function decide(approve) {
  $("approve").hidden = true;
  $("planActions").hidden = true;
  await post("/api/browser/approve", { approve });
  refreshBrowser();
}
$("approveYes").onclick = () => decide(true);
$("approveNo").onclick = () => decide(false);
$("planNo").onclick = () => decide(false);
$("planStart").onclick = async () => {
  if (editingPlan) await savePlan();
  decide(true);
};

async function savePlan() {
  const steps = $("planText").value.split("\n").map((l) => l.replace(/^\s*\d+[.)]\s*/, "").trim()).filter(Boolean);
  editingPlan = false;
  $("planText").hidden = true;
  if (steps.length) await post("/api/browser/plan", { steps });
}
$("planEditBtn").onclick = async () => {
  if (editingPlan) { await savePlan(); refreshBrowser(); return; }
  editingPlan = true;
  $("planText").value = (lastStatus?.plan || []).map((p) => p.text).join("\n");
  $("planText").hidden = false;
  $("planText").focus();
  $("planEditBtn").textContent = "Save plan";
};

$("bStop").onclick = async () => { await post("/api/browser/stop"); refreshBrowser(); };
$("bWindow").onclick = async () => {
  const show = $("bWindow").getAttribute("aria-pressed") !== "true";
  $("bWindow").disabled = true;
  const r = await post("/api/browser/visible", { visible: show });
  const j = await r.json().catch(() => ({}));
  $("bWindow").disabled = false;
  if (j.status && !["visible", "hidden"].includes(j.status)) add("event", j.status);
  else if (show) add("event", "The browser window is open on this computer. Sign in to the sites you want, then hide it.", "app");
  refreshBrowser();
};

$("taskForm").onsubmit = async (e) => {
  e.preventDefault();
  const goal = $("task").value.trim();
  if (!goal) return;
  $("task").value = "";
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission().catch(() => {});
  const r = await post("/api/browser/task", { goal, plan_first: $("planFirst").checked });
  const j = await r.json().catch(() => ({}));
  if (j.status?.startsWith("queued")) add("event", `Queued: ${goal}`, "app");
  refreshBrowser();
};

$("urlForm").onsubmit = async (e) => {
  e.preventDefault();
  const url = $("url").value.trim();
  if (!url) return;
  $("url").blur();
  const r = await post("/api/browser/open", { url });
  const j = await r.json().catch(() => ({}));
  if (j.error) add("event", j.error, "barge");
  refreshBrowser();
};

$("copyAnswer").onclick = () => {
  const text = [$("bSummary").textContent, $("bAnswer").textContent].filter(Boolean).join("\n\n");
  navigator.clipboard?.writeText(text).then(() => {
    $("copyAnswer").textContent = "Copied";
    setTimeout(() => ($("copyAnswer").textContent = "Copy"), 1500);
  });
};

const AUDIT_TEXT = {
  task_start: (e) => `Started: ${e.goal}`,
  plan: (e) => `Planned ${e.plan.length} step${e.plan.length === 1 ? "" : "s"}`,
  plan_edited: () => "You edited the plan",
  plan_decision: (e) => (e.approved ? "You started the plan" : "You cancelled the plan"),
  action: (e) => `${describe(e.action)}${e.result !== "ok" ? `: ${e.result}` : ""}`,
  approval: (e) => `${e.approved ? "You approved" : "Declined"}: ${e.description}`,
  task_end: (e) => `${e.state === "done" ? "Finished" : e.state === "failed" ? "Failed" : "Stopped"}: ${e.summary}`,
  task_stopped: () => "You stopped the task",
};

async function loadAudit() {
  let entries = [];
  try {
    const r = await api("/api/browser/audit");
    if (r.ok) entries = (await r.json()).entries || [];
  } catch {}
  const list = $("audit");
  list.innerHTML = "";
  if (!entries.length) list.innerHTML = "<li>Nothing yet. Every action, approval and result will be listed here.</li>";
  for (const e of entries) {
    const li = document.createElement("li");
    const t = document.createElement("time");
    t.textContent = e.t.slice(5, 16);
    li.append(t, document.createTextNode((AUDIT_TEXT[e.kind] || ((x) => x.kind))(e)));
    if (e.kind === "approval") li.className = e.approved ? "yes-t" : "no";
    if (e.kind === "action" && /^(failed|blocked|declined)/.test(e.result)) li.className = "no";
    list.append(li);
  }
}
function showTab(audit) {
  $("tabSteps").setAttribute("aria-selected", String(!audit));
  $("tabAudit").setAttribute("aria-selected", String(audit));
  $("bSteps").hidden = audit;
  $("audit").hidden = !audit;
  if (audit) loadAudit();
}
$("tabSteps").onclick = () => showTab(false);
$("tabAudit").onclick = () => showTab(true);

for (const b of document.querySelectorAll(".langs button")) {
  b.onclick = () => {
    $("lang").value = b.dataset.lang;
    for (const o of document.querySelectorAll(".langs button")) o.setAttribute("aria-pressed", String(o === b));
  };
}

// -- typed messages during a call -------------------------------------------------------------------
$("sayForm").onsubmit = (e) => {
  e.preventDefault();
  const text = $("say").value.trim();
  if (!text || ws?.readyState !== 1) return;
  $("say").value = "";
  if (interimEl) { interimEl.classList.remove("interim"); interimEl = null; }
  add("user", text);
  ws.send(JSON.stringify({ type: "typed", text }));
};

async function loadResume() {
  try {
    const r = await api("/api/resume");
    if (r.ok) $("resumeName").textContent = `Your details: ${(await r.json()).name}`;
  } catch {}
}
$("resumeFile").onchange = async () => {
  const f = $("resumeFile").files[0];
  if (!f) return;
  $("resumeName").textContent = `Uploading ${f.name}…`;
  const r = await api(`/api/resume?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
  const j = await r.json();
  $("resumeName").textContent = r.ok ? `Your details: ${j.name}` : j.error;
  $("resumeFile").value = "";
};
