"use strict";
// Lilly Beyond Talk: Altersabfrage, Szenenwahl mit Freischaltung, Anruf per WebRTC, Untertitel, Zeitlimit.
const $ = id => document.getElementById(id);
const out = $("out");
let ready = false, ageOk = false, peer = null, channel = null, stream = null, pcId = null;
let heartbeat = null, countdown = null, endsAt = 0, ending = false, maxSeconds = 300;
let scenes = [], currentScene = null, minSeconds = 60;

function say(text) { $("status").textContent = text; }
function showError(text) { const e = $("error"); e.textContent = text; e.hidden = !text; }
function label(id, text) { $(id).querySelector(".label").textContent = text; }
function portrait(state) {
  $("portrait").dataset.state = state;
  $("callview").dataset.state = state;
  if (state === "idle") stopLoop(); else startLoop();
}

// ---------- Video-Schleife (optional, fällt still auf das Standbild zurück) ----------
const loop = $("loop");
let loopMissing = false;
loop.muted = true;
loop.addEventListener("playing", () => loop.classList.add("on"));
loop.addEventListener("error", () => { loopMissing = true; loop.classList.remove("on"); });
function startLoop() {
  if (loopMissing || !currentScene || !loop.paused) return;
  if (!loop.getAttribute("src")) loop.src = `/static/scenes/${currentScene}.mp4`;
  loop.play().catch(() => { loop.classList.remove("on"); });
}
function stopLoop() {
  loop.pause();
  loop.classList.remove("on");
  try { loop.currentTime = 0; } catch { /* noch nichts geladen */ }
}
function resetLoop(id) {
  stopLoop();
  loopMissing = false;
  loop.removeAttribute("src");
  if (id) loop.poster = `/static/scenes/${id}.jpg`; else loop.removeAttribute("poster");
  loop.load();
}

async function post(path, body, timeout = 20000) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body), signal: AbortSignal.timeout(timeout)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Lilly ist gerade nicht erreichbar.");
  return data;
}

async function pollStatus() {
  try {
    const s = await (await fetch("/api/status", {cache: "no-store"})).json();
    maxSeconds = s.max_seconds || maxSeconds;
    $("gate-minutes").textContent = Math.round(maxSeconds / 60);
    ready = s.ready;
    if (s.llm_local === false) {
      $("gate-privacy").textContent = "Was du sagst, wird auf unserem Rechner in Text umgewandelt. Dieser Text geht für Lillys Antwort an " +
        (s.llm_provider || "einen externen KI-Dienst") + ". Aufnahmen werden nicht gespeichert. Gespeichert wird nur, welche Szenen du schon gespielt hast (über ein Cookie).";
    }
    if (!peer) {
      say(s.ready ? (s.busy ? "Lilly ist gerade in einem Gespräch." : "Lilly ist da. Setz dich zu ihr!") : s.stage);
      $("call").disabled = !s.ready || !ageOk || !currentScene;
    }
  } catch { if (!peer) say("Keine Verbindung zum Server."); }
  setTimeout(pollStatus, peer ? 10000 : 3000);
}

function receive(event) {
  let m; try { m = JSON.parse(event.data); } catch { return; }
  if (m.type === "state") {
    if (m.state === "listening") { portrait("listening"); say("Lilly hört zu …"); }
    if (m.state === "speaking") { portrait("speaking"); say("Lilly spricht …"); }
    if (m.state === "responding") { portrait("listening"); say("Lilly überlegt …"); }
  }
  if (m.type === "partial" && m.text) $("you").textContent = m.text;
  if (m.type === "transcript") {
    if (m.role === "user" && m.text) $("you").textContent = m.text;
    if (m.role === "assistant" && m.text) $("lilly").textContent = m.text;
  }
  if (m.type === "limit_reached") { say("Die Zeit ist um. Danke fürs Quatschen!"); void hangUp(); }
  if (m.type === "error") showError(m.message || "Da ist etwas schiefgegangen.");
}

async function waitIce(pc) {
  if (pc.iceGatheringState === "complete") return;
  await new Promise((resolve, reject) => {
    const t = setTimeout(() => { done(); reject(new Error("Verbindungsaufbau dauert zu lange.")); }, 10000);
    function done() { clearTimeout(t); pc.removeEventListener("icegatheringstatechange", check); }
    function check() { if (pc.iceGatheringState === "complete") { done(); resolve(); } }
    pc.addEventListener("icegatheringstatechange", check);
    check();
  });
}

function startCountdown(seconds) {
  endsAt = Date.now() + seconds * 1000;
  $("timer").hidden = false;
  const tick = () => {
    const left = Math.max(0, Math.round((endsAt - Date.now()) / 1000));
    $("timer").textContent = `Noch ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  };
  tick();
  countdown = setInterval(tick, 1000);
}

async function call() {
  showError("");
  $("call").disabled = true;
  say("Mikrofon wird verbunden …");
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Dein Browser erlaubt hier kein Mikrofon (HTTPS nötig).");
    stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}, video: false});
    const pc = new RTCPeerConnection({iceServers: []});
    peer = pc;
    stream.getTracks().forEach(t => pc.addTrack(t, stream));
    channel = pc.createDataChannel("lilly");
    channel.onmessage = receive;
    channel.onopen = () => { heartbeat = setInterval(() => { if (channel?.readyState === "open") channel.send("ping"); }, 1000); };
    pc.ontrack = e => { out.srcObject = e.streams[0] || new MediaStream([e.track]); out.play().catch(() => showError("Bitte Tonwiedergabe im Browser erlauben.")); };
    pc.onconnectionstatechange = () => {
      if (peer !== pc) return;
      if (pc.connectionState === "connected") { say("Verbunden. Lilly meldet sich …"); portrait("listening"); }
      if (["failed", "disconnected", "closed"].includes(pc.connectionState) && !ending) { say("Verbindung beendet."); void hangUp(); }
    };
    say("Du gehst zu Lilly …");
    await pc.setLocalDescription(await pc.createOffer());
    await waitIce(pc);
    const answer = await post("/api/offer", {sdp: pc.localDescription.sdp, type: "offer", age_confirmed: ageOk, scene: currentScene});
    pcId = answer.pc_id;
    await pc.setRemoteDescription({sdp: answer.sdp, type: answer.type});
    startCountdown(answer.max_seconds || maxSeconds);
    label("call", "Gespräch beenden");
    $("call").dataset.active = "true";
    $("call").disabled = false;
    $("mute").hidden = false;
    $("back").hidden = true;
  } catch (error) {
    await hangUp(true);
    const known = {NotAllowedError: "Ohne Mikrofon kann Lilly dich nicht hören. Bitte im Browser erlauben.",
      NotFoundError: "Kein Mikrofon gefunden.", NotReadableError: "Das Mikrofon ist gerade belegt."};
    showError(known[error.name] || error.message);
  }
}

async function hangUp(quiet = false) {
  if (ending) return;
  ending = true;
  const id = pcId; pcId = null;
  clearInterval(heartbeat); heartbeat = null;
  clearInterval(countdown); countdown = null;
  $("timer").hidden = true;
  stream?.getTracks().forEach(t => t.stop()); stream = null;
  channel?.close(); channel = null;
  peer?.close(); peer = null;
  out.pause(); out.srcObject = null;
  portrait("idle");
  label("call", "Mit Lilly reden");
  $("call").dataset.active = "false";
  $("mute").hidden = true;
  $("mute").setAttribute("aria-pressed", "false");
  label("mute", "Mikro stumm");
  try { if (id) await post("/api/disconnect", {pc_id: id}, 10000); } catch { /* Server räumt selbst auf */ }
  ending = false;
  $("back").hidden = false;
  $("call").disabled = !ready || !ageOk;
  setTimeout(loadScenes, 800); // Server trägt die gespielte Szene beim Auflegen ein
  if (!quiet && !$("status").textContent.startsWith("Die Zeit")) say("Aufgelegt. Bis bald!");
}

$("call").addEventListener("click", () => { if (peer) void hangUp(); else void call(); });
$("mute").addEventListener("click", () => {
  const muted = $("mute").getAttribute("aria-pressed") !== "true";
  stream?.getAudioTracks().forEach(t => { t.enabled = !muted; });
  $("mute").setAttribute("aria-pressed", String(muted));
  label("mute", muted ? "Mikro an" : "Mikro stumm");
});

// ---------- Szenen ----------
async function loadScenes() {
  try {
    const r = await fetch("/api/scenes", {cache: "no-store", credentials: "same-origin"});
    const data = await r.json();
    scenes = data.scenes; minSeconds = data.min_seconds || minSeconds;
    renderScenes();
  } catch { showError("Szenen konnten nicht geladen werden."); }
}

function sceneMeta(sc, index) {
  if (sc.played) return "Gespielt · nochmal treffen";
  if (!sc.unlocked) return `Frei, wenn du „${scenes[index - 1].title}“ gespielt hast`;
  if (sc.needs_payment) return "Premium · jetzt freischalten (Simulation)";
  return sc.teaser;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function renderScenes() {
  const list = $("scenes");
  list.replaceChildren(...scenes.map((sc, i) => {
    const li = document.createElement("li");
    const b = el("button", "scene");
    b.type = "button";
    b.dataset.id = sc.id;
    const locked = !sc.unlocked;
    if (locked) b.setAttribute("aria-disabled", "true");
    if (sc.premium) b.classList.add("is-premium");
    const art = el("img", `art ${sc.id}`); art.alt = "";
    art.src = `/static/scenes/${sc.id}.jpg`; art.width = 900; art.height = 1152;
    art.loading = i < 2 ? "eager" : "lazy"; art.decoding = "async";
    const tags = el("span", "tags");
    tags.append(el("span", "num", `Szene ${i + 1}`));
    if (sc.premium) tags.append(el("span", "badge premium", "18+ Premium"));
    if (sc.played) { const t = el("span", "badge done", "gespielt"); t.prepend(icon("check")); tags.append(t); }
    if (locked) { const t = el("span", "badge locked", "gesperrt"); t.prepend(icon("lock")); tags.append(t); }
    const body = el("span", "body");
    body.append(el("span", "name", sc.title), el("span", "meta", sceneMeta(sc, i)));
    if (!locked) {
      const cta = el("span", "cta", sc.needs_payment ? "Freischalten" : sc.played ? "Nochmal treffen" : "Treffen");
      cta.prepend(icon(sc.needs_payment ? "lock" : "phone"));
      body.append(cta);
    }
    b.append(art, tags, body);
    if (locked) { const l = el("span", "lock"); l.append(icon("lock")); b.append(l); }
    b.addEventListener("click", () => chooseScene(sc));
    li.append(b);
    return li;
  }));
}

function chooseScene(sc) {
  if (!ageOk) { gate.showModal(); return; }
  if (!sc.unlocked) { showError(`Spiel erst „${scenes[scenes.indexOf(sc) - 1].title}“ mindestens eine Minute.`); return; }
  if (sc.needs_payment) { pending = sc; $("pay").showModal(); return; }
  openScene(sc);
}

function openScene(sc) {
  showError("");
  currentScene = sc.id;
  $("scene-title").textContent = sc.title;
  $("scene-teaser").textContent = sc.teaser || "";
  const img = $("portrait").querySelector("img");
  img.src = `/static/scenes/${sc.id}.jpg`; img.alt = `Lilly – ${sc.title}`; img.width = 900; img.height = 1152;
  resetLoop(sc.id);
  $("you").textContent = ""; $("lilly").textContent = "";
  $("picker").hidden = true;
  $("callview").hidden = false;
  document.body.dataset.view = "call";
  window.scrollTo(0, 0);
  $("call").disabled = !ready;
  $("call").focus();
}

function backToScenes() {
  if (peer) return;
  currentScene = null;
  resetLoop(null);
  $("callview").hidden = true;
  $("picker").hidden = false;
  document.body.dataset.view = "picker";
  void loadScenes();
}

let pending = null;
$("pay").addEventListener("close", async () => {
  const choice = $("pay").returnValue; $("pay").returnValue = "";
  const sc = pending; pending = null;
  if (choice !== "pay" || !sc) return;
  try {
    const data = await post("/api/pay-simulated", {});
    scenes = data.scenes; renderScenes();
    openScene(scenes.find(s => s.id === sc.id));
  } catch (error) { showError(error.message); }
});
$("back").addEventListener("click", backToScenes);

const gate = $("gate");
gate.addEventListener("cancel", e => e.preventDefault()); // ohne Entscheidung kein Weiter
gate.addEventListener("close", () => {
  const choice = gate.returnValue; gate.returnValue = "";
  if (choice === "yes") { ageOk = true; $("call").disabled = !ready; }
  else { ageOk = false; $("call").disabled = true; say("Lilly Beyond Talk ist erst ab 18. Schau gern ihre Videos auf YouTube!"); }
});
$("info").addEventListener("click", () => gate.showModal());
gate.showModal();
pollStatus();
loadScenes();
