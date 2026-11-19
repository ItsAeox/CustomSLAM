// Blind user study (claude/STUDY.md): shared state, the response store, and the guided task overlay used by the three
// condition pages (WebVIO = ../ar.js in study mode, AlvaAR = qm.html, WebXR = tz.html). The pages never show which
// tracker runs; they are opened under neutral names and the order comes from the participant's Latin-square row.

export const STORE_URL = 'https://script.google.com/macros/s/AKfycby0X8_1nNfMOQZRwov2GnZAOAqexDkBV_Z_WuZBxbes9d2EguP-zbpJT_C7SfUMrRk/exec';  // apps_script.gs web app (user's sheet, 7 Nov)
export const CONDITIONS = { webvio: 'kx.html', alvaar: 'qm.html', webxr: 'tz.html' };  // study/<page>
export const ORDERS = [['webvio', 'alvaar', 'webxr'], ['alvaar', 'webxr', 'webvio'], ['webxr', 'webvio', 'alvaar'],
                       ['webvio', 'webxr', 'alvaar'], ['alvaar', 'webvio', 'webxr'], ['webxr', 'alvaar', 'webvio']];
export const WEBVIO_PRESET = 'beta';
// Filled in by the researcher before the study opens. While either is empty the consent page says the study is not
// open and its Start button stays off (test sessions, ?test=1, are not blocked).
export const CONTACT = 'Gayathra Dissanayake, gayathradissanayake@gmail.com';  // researcher name and e-mail shown to participants
export const ETHICS = '';   // ethics approval: committee and reference number
const KEY = 'webvio-study-v1';

export function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || null; } catch { return null; }
}
export function save(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage blocked: the session still works in memory */ }
}
export function newParticipant() {
  const id = Array.from(crypto.getRandomValues(new Uint8Array(6)), (b) => b.toString(16).padStart(2, '0')).join('');
  return { id, started: new Date().toISOString(), stage: 'consent', order: null, orderSource: null, k: 0, answers: {}, runs: [] };
}

// Each stage is sent as soon as it is done (partial sessions count). Simple text/plain POST: no preflight, and the
// Apps Script answer comes back with CORS, which the cross-origin-isolated pages accept.
export async function send(s, stage, data) {
  const row = { id: s.id, stage: (s.test ? 'TEST-' : '') + stage, t: new Date().toISOString(), ua: navigator.userAgent, data };
  s.sent = s.sent || [];
  s.sent.push({ stage, t: row.t, ok: false });
  save(s);
  if (!STORE_URL) return false;
  try {
    const r = await fetch(STORE_URL, { method: 'POST', mode: 'cors', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
                                       body: JSON.stringify(row) });
    const ok = r.ok;
    s.sent[s.sent.length - 1].ok = ok; save(s);
    return ok;
  } catch {
    try { return navigator.sendBeacon(STORE_URL, new Blob([JSON.stringify(row)], { type: 'text/plain' })); } catch { return false; }
  }
}

// Balanced order: the store hands out Latin-square rows in turn (6 orders); offline, a random row (logged as such).
export async function assignOrder(test = false) {
  if (STORE_URL && !test) {  // test sessions do not use up the balanced order list
    try {
      const r = await fetch(`${STORE_URL}?action=order`, { mode: 'cors' });
      const j = await r.json();
      if (Number.isInteger(j.row)) return { row: j.row % ORDERS.length, source: 'store' };
    } catch { /* fall through */ }
  }
  return { row: Math.floor(Math.random() * ORDERS.length), source: 'random' };
}

// ------------------------------------------------------------------------------------------- guided task overlay
// The condition page calls overlay.frame(tracking) once per processed frame and overlay.placed() when the object is
// placed; it provides adapter.objectScreen() -> {x, y, visible} (client pixels) or null, and optional
// adapter.brightness() -> 0..255 or null. Steps follow claude/STUDY.md §2.
export const STEPS = [
  { key: 'start', text: 'Point the camera at a table or the floor and move the phone slowly sideways until it says “Ready”.' },
  { key: 'place', text: 'Put a small real object (a coin, a key, a cup) on the table or floor. Tap the screen right next to it to place the virtual plant.' },
  { key: 'still', text: 'Hold the phone still and look at both objects.', seconds: 5 },
  { key: 'around', text: 'Walk slowly half a circle around the objects, keeping them in view.', seconds: 20, early: 15 },
  { key: 'fast', text: 'Turn quickly away and back, twice.', seconds: 6, early: 4 },
  { key: 'away', text: 'Turn away or walk 3–4 metres away for about 10 seconds, then come back to where you started.', seconds: 25, early: 10 },
  { key: 'check', text: 'Tap exactly on your REAL object.' },
];

export const SKIP_S = 90;  // a participant stuck on 'start' or 'place' this long can skip the version (recorded as failed)

export class Overlay {
  constructor(adapter, onDone) {
    this.adapter = adapter; this.onDone = onDone;
    this.t0 = performance.now();
    this.log = { frames: 0, firstTrackS: null, losses: [], lostSince: null, readyS: null, placedS: null, steps: [],
                 brightness: [], check: null, screen: { w: innerWidth, h: innerHeight, dpr: devicePixelRatio } };
    this.step = 0; this.stepT = performance.now(); this.trackedRun = 0; this.lastT = null;
    this.el = document.createElement('div');
    this.el.id = 'studyOverlay';
    this.el.innerHTML = `<style>
      #studyOverlay { position: fixed; left: 0; right: 0; bottom: 0; z-index: 50; padding: 14px 16px max(16px, env(safe-area-inset-bottom));
        background: rgba(0,0,0,.62); color: #fff; font: 16px/1.4 system-ui, sans-serif; }
      #studyOverlay .row { display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-top: 10px; }
      #studyOverlay [hidden] { display: none !important; }
      #studyOverlay button { font: 600 16px system-ui; padding: 10px 18px; border-radius: 10px; border: 0; background: #22c55e; color: #000; }
      #studyOverlay button:disabled { background: #555; color: #aaa; }
      #studyOverlay .muted { color: rgba(255,255,255,.7); font-size: 13px; }
      #studyBadge { position: fixed; top: max(12px, env(safe-area-inset-top)); left: 50%; transform: translateX(-50%); z-index: 50;
        padding: 6px 14px; border-radius: 16px; background: rgba(0,0,0,.6); color: #fff; font: 600 14px system-ui; }
    </style>
    <div id="studyText"></div>
    <div class="row"><span class="muted" id="studyInfo"></span><button id="studyNext" disabled>Next</button></div>
    <div class="row" id="studySkipRow" hidden><span class="muted">Not working after ${SKIP_S} seconds?</span>
      <button id="studySkip" style="background:#666;color:#fff">Skip this version</button></div>`;
    document.body.append(this.el);
    this.badge = Object.assign(document.createElement('div'), { id: 'studyBadge' });
    document.body.append(this.badge);
    for (const ev of ['beforexrselect']) this.el.addEventListener(ev, (e) => e.preventDefault());  // WebXR: buttons are not taps
    this.el.querySelector('#studyNext').onclick = () => this.next();
    this.el.querySelector('#studySkip').onclick = () => {
      const st = STEPS[this.step];
      this.log.steps.push({ key: st.key, s: +((performance.now() - this.stepT) / 1000).toFixed(2), skipped: true });
      this.fail(`skipped at ${st.key}`, 'You skipped this version. Please continue.');
    };
    this.show();
    this.timer = setInterval(() => this.tick(), 250);
    this.brightTimer = setInterval(() => {
      const b = this.adapter.brightness ? this.adapter.brightness() : null;
      if (b != null) this.log.brightness.push(Math.round(b));
    }, 1000);
  }
  now() { return (performance.now() - this.t0) / 1000; }
  setBadge(s) { this.badge.textContent = s; this.badge.hidden = !s; }
  show() {
    const s = STEPS[this.step];
    this.el.querySelector('#studyText').textContent = `${this.step + 1} / ${STEPS.length}  ${s.text}`;
    this.stepT = performance.now();
    this.update();
  }
  update() {
    const s = STEPS[this.step], el = (performance.now() - this.stepT) / 1000;
    let ok = false, info = '';
    if (s.key === 'start') { ok = this.log.readyS != null; info = ok ? 'Ready' : 'Starting…'; }
    else if (s.key === 'place') { ok = this.log.placedS != null; info = ok ? 'Placed' : ''; }
    else if (s.key === 'check') { ok = false; info = this.log.check ? '' : 'Tap your real object'; }
    else { ok = el >= (s.early ?? s.seconds); info = `${Math.max(0, Math.ceil(s.seconds - el))} s`; }
    this.el.querySelector('#studyNext').disabled = !ok;
    // the skip clock starts at the step or at the first camera frame, whichever is later (permission prompts do not count)
    const skipEl = (performance.now() - Math.max(this.stepT, this.firstFrameT ?? this.stepT)) / 1000;
    this.el.querySelector('#studySkipRow').hidden = ok || !(s.key === 'start' || s.key === 'place') || skipEl < SKIP_S;
    this.el.querySelector('#studyInfo').textContent = info;
    this.setBadge(s.key === 'start' && this.log.readyS != null ? 'Ready' : '');
  }
  tick() { this.update(); }
  next() {
    const s = STEPS[this.step];
    this.log.steps.push({ key: s.key, s: +((performance.now() - this.stepT) / 1000).toFixed(2) });
    if (this.step < STEPS.length - 1) { this.step++; this.show(); }
  }
  // per processed frame
  frame(tracking) {
    const t = this.now();
    if (this.log.frames === 0) this.firstFrameT = performance.now();
    this.log.frames++;
    if (tracking) {
      if (this.log.firstTrackS == null) this.log.firstTrackS = +t.toFixed(2);
      if (this.log.lostSince != null) { this.log.losses.push([+this.log.lostSince.toFixed(2), +(t - this.log.lostSince).toFixed(2), STEPS[this.step].key]); this.log.lostSince = null; }
      this.trackedRun = this.lastT == null ? 0 : this.trackedRun + (t - this.lastT);
      if (this.log.readyS == null && this.trackedRun >= 1) { this.log.readyS = +t.toFixed(2); this.update(); }
    } else {
      this.trackedRun = 0;
      if (this.log.firstTrackS != null && this.log.lostSince == null) this.log.lostSince = t;
    }
    this.lastT = t;
  }
  placed() {
    if (this.log.placedS == null) { this.log.placedS = +this.now().toFixed(2); this.update(); }
  }
  // the page forwards taps here first: true = consumed (return check), so it does not place the object again
  tap(clientX, clientY) {
    const key = STEPS[this.step].key;
    if (key === 'start' || key === 'place') return false;  // placing (again) is allowed until the object is placed
    if (key !== 'check' || this.log.check) return true;     // later taps do not move the object
    const o = this.adapter.objectScreen ? this.adapter.objectScreen() : null;
    const diag = Math.hypot(innerWidth, innerHeight);
    this.log.check = { tap: [Math.round(clientX), Math.round(clientY)], object: o ? [Math.round(o.x), Math.round(o.y)] : null,
                       visible: !!(o && o.visible), offsetPx: o ? +Math.hypot(o.x - clientX, o.y - clientY).toFixed(1) : null,
                       offsetDiag: o ? +(Math.hypot(o.x - clientX, o.y - clientY) / diag).toFixed(4) : null, t: +this.now().toFixed(2) };
    this.finish();
    return true;
  }
  finish() {
    if (this.log.lostSince != null) { this.log.losses.push([+this.log.lostSince.toFixed(2), +(this.now() - this.log.lostSince).toFixed(2), 'open']); this.log.lostSince = null; }
    this.log.steps.push({ key: 'check', s: +((performance.now() - this.stepT) / 1000).toFixed(2) });
    this.log.totalS = +this.now().toFixed(2);
    this.log.fps = this.log.totalS > 0 ? +(this.log.frames / this.log.totalS).toFixed(1) : null;
    clearInterval(this.timer); clearInterval(this.brightTimer);
    this.el.querySelector('#studyText').textContent = 'Done. Thank you!';
    this.el.querySelector('#studyInfo').textContent = '';
    const b = this.el.querySelector('#studyNext');
    b.textContent = 'Continue'; b.disabled = false;
    b.onclick = () => this.onDone(this.log);
  }
  fail(reason, text = 'This version could not start on your phone. Please continue.') {  // recorded as a failure of this condition
    if (this.log.lostSince != null) { this.log.losses.push([+this.log.lostSince.toFixed(2), +(this.now() - this.log.lostSince).toFixed(2), 'open']); this.log.lostSince = null; }
    this.log.failed = reason; this.log.totalS = +this.now().toFixed(2);
    this.log.fps = this.log.totalS > 0 ? +(this.log.frames / this.log.totalS).toFixed(1) : null;
    clearInterval(this.timer); clearInterval(this.brightTimer);
    this.el.querySelector('#studySkipRow').hidden = true;
    this.el.querySelector('#studyInfo').textContent = '';
    this.el.querySelector('#studyText').textContent = text;
    const b = this.el.querySelector('#studyNext');
    b.textContent = 'Continue'; b.disabled = false;
    b.onclick = () => this.onDone(this.log);
  }
}

// Called by a condition page when its task is done: keeps the log, sends it, returns to the questionnaire.
export async function finishCondition(condition, log) {
  const s = load();
  const back = new URL('./index.html', import.meta.url).href;  // study/index.html, whatever the page's base
  if (!s) { location.href = back; return; }
  s.runs[s.k] = { condition, log };
  s.stage = `q${s.k}`;
  save(s);
  await send(s, `run${s.k}`, { condition, log });
  location.href = back;
}

// Mean brightness (0..255) of a canvas, from a 32 x 32 downscale (cheap; nothing leaves the phone)
let tiny = null;
export function canvasBrightness(src) {
  if (!src || !src.width) return null;
  tiny ??= Object.assign(document.createElement('canvas'), { width: 32, height: 32 });
  const c = tiny.getContext('2d', { willReadFrequently: true });
  c.drawImage(src, 0, 0, 32, 32);
  const d = c.getImageData(0, 0, 32, 32).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  return s / (d.length / 4);
}

// Same first screen for every version (WebXR needs a tap to start its session, so all three wait for one)
export function startGate(k, onStart) {
  const g = document.createElement('div');
  g.style.cssText = 'position:fixed;inset:0;z-index:60;display:flex;flex-direction:column;align-items:center;justify-content:center;' +
                    'gap:18px;background:#111;color:#fff;font:18px/1.4 system-ui,sans-serif;text-align:center;padding:24px';
  g.innerHTML = `<div style="font:600 24px system-ui">Version ${k + 1}</div><div>Tap Start, then allow the camera if the phone asks.</div>` +
                `<button style="font:600 18px system-ui;padding:14px 40px;border:0;border-radius:12px;background:#22c55e;color:#000">Start</button>`;
  document.body.append(g);
  g.querySelector('button').onclick = async () => { g.remove(); await onStart(); };
}
export const versionIndex = () => (load() || { k: 0 }).k;
