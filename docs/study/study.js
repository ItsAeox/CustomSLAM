// Study flow (claude/STUDY.md): consent -> background -> environment -> device check -> brief -> 3 x (version k, its
// questionnaire) -> comparison -> thank you. State in localStorage (common.js), each stage sent when done.
import { CONDITIONS, CONTACT, ETHICS, ORDERS, WEBVIO_PRESET, assignOrder, load, newParticipant, save, send } from './common.js';

const app = document.getElementById('app');
let s = load();
if (!s || new URLSearchParams(location.search).get('new') === '1') { s = newParticipant(); save(s); }
if (new URLSearchParams(location.search).get('test') === '1' && !s.test) { s.test = true; save(s); }  // flow test: every row TEST-

const DEV = location.pathname.startsWith('/web/');
const ROOT = new URL('../', location.href);
const VOCAB = DEV ? new URL('/vocab/ORBvoc.bin', location.href).href : new URL('vocab/ORBvoc.bin.gz', ROOT).href;

const T_ITEMS = [
  ['T1', 'The app was ready for placing the object quickly.'],
  ['T2', 'The virtual object appeared where I tapped.'],
  ['T3', 'When I held the phone still, the virtual object stayed still (no shaking or jitter).'],
  ['T4', 'When I walked around, the virtual object stayed fixed to the same real spot.'],
  ['T5', 'The virtual object kept a believable size as I moved closer or further away.'],
  ['T6', 'After the quick turns, the virtual object was still in the right place.'],
  ['T8', 'When it was lost, it came back quickly.', 'na'],
  ['T9', 'When I came back, the virtual object was in the same place as before.'],
  ['T10', 'The camera view and the virtual object moved smoothly (no lag or stutter).'],
  ['T11', 'The app worked well in the lighting I was in.'],
  ['T12', 'Overall, the virtual object looked as if it were really standing in the room.'],
];
const UEQS = [['obstructive', 'supportive'], ['complicated', 'easy'], ['inefficient', 'efficient'], ['confusing', 'clear'],
              ['boring', 'exciting'], ['not interesting', 'interesting'], ['conventional', 'inventive'], ['usual', 'leading edge']];

const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const likert = (name, q, lo = 'Strongly disagree', hi = 'Strongly agree', na = false) => `
  <div class="card"><p class="q">${esc(q)}</p>
    <div class="scale">${[1, 2, 3, 4, 5, 6, 7].map((v) => `<label><input type="radio" name="${name}" value="${v}" required>${v}</label>`).join('')}</div>
    <div class="ends"><span>${esc(lo)}</span><span>${esc(hi)}</span></div>
    ${na ? `<div class="opts"><label><input type="radio" name="${name}" value="na">It was never lost</label></div>` : ''}
  </div>`;
const choice = (name, q, options) => `
  <div class="card"><p class="q">${esc(q)}</p><div class="opts">
    ${options.map(([v, l]) => `<label><input type="radio" name="${name}" value="${esc(v)}" required>${esc(l)}</label>`).join('')}
  </div></div>`;
const versions = (name, q) => choice(name, q, [['1', 'Version 1'], ['2', 'Version 2'], ['3', 'Version 3'], ['none', 'No difference']]);

function form(html, onSubmit, button = 'Continue') {
  app.innerHTML = `<form id="f">${html}<p class="err" id="err"></p><button class="primary" type="submit">${esc(button)}</button></form>`;
  window.scrollTo(0, 0);
  const f = document.getElementById('f');
  f.onsubmit = async (e) => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(f).entries());
    const missing = [...new Set([...f.querySelectorAll('input[required]')].map((i) => i.name))].filter((n) => !(n in d));
    if (missing.length) { document.getElementById('err').textContent = 'Please answer every question.'; return; }
    f.querySelector('button').disabled = true;
    await onSubmit(d);
  };
}
const go = (stage) => { s.stage = stage; save(s); render(); };

function render() {
  const st = s.stage;
  if (st === 'consent') return consent();
  if (st === 'background') return background();
  if (st === 'environment') return environment();
  if (st === 'check') return check();
  if (st === 'brief') return brief();
  if (/^run\d$/.test(st)) return runIntro(Number(st[3]));
  if (/^q\d$/.test(st)) return questionnaire(Number(st[1]));
  if (st === 'compare') return compare();
  if (st === 'done' || st === 'ineligible') return done();
  return consent();
}

function consent() {
  app.innerHTML = `<h1>Comparing ways of placing virtual objects in the camera view</h1>
    <p>This study is part of a research project at the University of Colombo School of Computing, Sri Lanka. You will try
       three versions of a web page that places a small virtual plant in your camera view, and answer a few questions
       after each. It takes about 20 minutes.</p>
    <h2>What we record</h2>
    <p>Your answers, and measurements made on your phone during the task: whether tracking was working, how long it
       took, the image brightness, where you tapped, and your phone and browser type. <b>No camera images, video,
       location, name or e-mail are recorded or sent.</b> The camera image is processed on your phone only.</p>
    <h2>Your rights</h2>
    <p>Taking part is voluntary. You can stop at any time by closing the page. Your answers are stored under a random
       code shown at the end; to have them deleted later, send that code to the researcher.</p>
    <h2>Safety</h2>
    <p>You will walk a few steps while looking at your phone. Use a clear space, away from stairs and traffic.</p>
    ${CONTACT && ETHICS ? `<p class="muted">Ethics approval: ${esc(ETHICS)}<br>Contact: ${esc(CONTACT)}</p>`
      : `<p class="card" style="border:2px solid #c33">This study is not open yet: the ethics reference and the researcher's contact
         details are still to be added.${s.test ? ' (Test session: you can continue.)' : ''}</p>`}
    <form id="f"><div class="card opts">
      <label><input type="checkbox" id="adult">I am 18 years or older.</label>
      <label><input type="checkbox" id="agree">I have read the information above and agree to take part.</label>
    </div><button class="primary" id="b" disabled>Start</button></form>`;
  const open = (CONTACT && ETHICS) || s.test;  // not open to participants until both are filled in (common.js)
  const upd = () => { document.getElementById('b').disabled = !(open && document.getElementById('adult').checked && document.getElementById('agree').checked); };
  document.getElementById('adult').onchange = upd; document.getElementById('agree').onchange = upd;
  document.getElementById('f').onsubmit = async (e) => {
    e.preventDefault();
    s.answers.consent = { adult: true, agree: true, t: new Date().toISOString() };
    await send(s, 'consent', s.answers.consent);
    preload();
    go('background');
  };
}

function background() {
  form(`<h1>About you</h1>
    ${choice('age', 'Your age', [['18-24', '18–24'], ['25-34', '25–34'], ['35-44', '35–44'], ['45-54', '45–54'], ['55+', '55 or older'], ['no', 'Prefer not to say']])}
    ${choice('arExp', 'Have you used augmented reality (AR) on a phone before?', [['never', 'Never'], ['few', 'A few times'], ['regular', 'Regularly']])}
    ${choice('expertise', 'Do you work or study in AR, SLAM or computer vision?', [['none', 'No'], ['some', 'Some knowledge'], ['pro', 'Yes, professionally or as a researcher']])}
    ${choice('hand', 'Which hand will hold the phone?', [['right', 'Right'], ['left', 'Left'], ['both', 'Both'], ['no', 'Prefer not to say']])}`,
  async (d) => { s.answers.background = d; await send(s, 'background', d); go('environment'); });
}

function environment() {
  form(`<h1>Where you are</h1>
    ${choice('place', 'Where are you doing this?', [['home', 'A room at home'], ['office', 'An office or lab'], ['outdoors', 'Outdoors'], ['other', 'Somewhere else']])}
    ${choice('light', 'How bright is it?', [['dim', 'Dim'], ['normal', 'Normal'], ['bright', 'Bright']])}
    ${choice('surface', 'The surface where you will put an object (table or floor) is…', [['plain', 'Plain (one colour)'], ['some', 'Has some texture or pattern'], ['busy', 'Has a busy pattern']])}`,
  async (d) => { s.answers.environment = d; await send(s, 'environment', d); go('check'); });
}

async function check() {
  app.innerHTML = `<h1>Checking your phone…</h1><p class="muted">One moment.</p>`;
  const r = { isolated: window.crossOriginIsolated, threads: navigator.hardwareConcurrency, xr: false, camera: !!navigator.mediaDevices?.getUserMedia };
  try { r.xr = !!(navigator.xr && await navigator.xr.isSessionSupported('immersive-ar')); } catch { r.xr = false; }
  if (new URLSearchParams(location.search).get('test') === '1' || s.test) { s.test = true; r.test = true; r.xr = true; }  // flow test on a PC: flagged in the data
  s.answers.device = r;
  if (!r.xr || !r.camera) {
    await send(s, 'ineligible', r);
    s.stage = 'ineligible'; save(s);
    app.innerHTML = `<h1>Sorry, this phone cannot run all three versions</h1>
      <p>The study needs an Android phone with Chrome and Google Play Services for AR (ARCore). Thank you for your time.</p>`;
    return;
  }
  if (!s.order) {
    const o = await assignOrder(!!s.test);
    s.order = ORDERS[o.row]; s.orderRow = o.row; s.orderSource = o.source;
  }
  await send(s, 'device', { ...r, orderRow: s.orderRow, orderSource: s.orderSource });
  go('brief');
}

function brief() {
  app.innerHTML = `<h1>What you will do</h1>
    <p>You will try <b>three versions</b>, one after the other. Each time you will:</p>
    <ol><li>move the phone slowly sideways until it says “Ready”;</li>
      <li>put a small real object (a coin, a key, a cup) on a table or the floor, and tap next to it to place a virtual plant;</li>
      <li>hold still for a few seconds, walk half a circle around them, then turn quickly away and back twice;</li>
      <li>turn away or walk 3–4 metres away for about 10 seconds, come back, and tap on your real object.</li></ol>
    <p>The page guides you step by step. After each version you answer a short set of questions.</p>
    <p><b>Please have a small object ready now.</b> Use a clear space, away from stairs and traffic.</p>
    <p class="muted">The versions may look the same; please judge each on how well the plant stays in its place.</p>
    <button class="primary" id="b">I am ready</button>`;
  document.getElementById('b').onclick = () => go('run0');
}

function runIntro(k) {
  app.innerHTML = `<h1>Version ${k + 1}</h1>
    <p>Start Version ${k + 1}. Allow the camera${s.order[k] === 'webxr' ? ' and AR' : ''} when the phone asks.</p>
    <button class="primary" id="b">Start Version ${k + 1}</button>`;
  document.getElementById('b').onclick = () => {
    s.k = k; save(s);
    const c = s.order[k];
    location.href = new URL(CONDITIONS[c] + (c === 'webvio' ? `?study=1&preset=${WEBVIO_PRESET}` : ''), location.href).href;
  };
}

function questionnaire(k) {
  const run = s.runs[k];
  const failed = run && run.log && run.log.failed;
  form(`<h1>Version ${k + 1}: your impressions</h1>
    ${failed ? `<p class="muted">${String(failed).startsWith('skipped') ? 'You skipped this version' : 'This version did not start on your phone'};
       please answer the questions you can, as you experienced it.</p>` : ''}
    <p class="muted">1 = strongly disagree, 7 = strongly agree.</p>
    ${T_ITEMS.slice(0, 6).map(([n, q, na]) => likert(n, q, undefined, undefined, na === 'na')).join('')}
    ${choice('T7', 'How often did the virtual object disappear, jump or slide away?', [['0', 'Never'], ['1', 'Once'], ['2-3', '2–3 times'], ['4+', '4 times or more']])}
    ${T_ITEMS.slice(6).map(([n, q, na]) => likert(n, q, undefined, undefined, na === 'na')).join('')}
    <h2>Your overall impression of Version ${k + 1}</h2>
    ${UEQS.map(([lo, hi], i) => likert(`U${i + 1}`, `${lo}  –  ${hi}`, lo, hi)).join('')}`,
  async (d) => {
    s.answers[`q${k}`] = d;
    await send(s, `q${k}`, { version: k + 1, condition: s.order[k], answers: d });
    go(k < 2 ? `run${k + 1}` : 'compare');
  });
}

function compare() {
  form(`<h1>Comparing the three versions</h1>
    ${versions('C1', 'Which version kept the virtual object in place best?')}
    ${versions('C2', 'Which version recovered best after losing the object?')}
    ${versions('C3', 'Which version had the object in the right place when you came back?')}
    ${versions('C4', 'Which version felt smoothest?')}
    ${versions('C5', 'Which version would you rather use?')}
    ${choice('C6', 'How sure are you about these answers?', [['1', '1 – not sure at all'], ['2', '2'], ['3', '3'], ['4', '4'], ['5', '5 – very sure']])}
    ${choice('B1', 'Did one version look or start differently from the others (for example an extra permission screen)?', [['no', 'No'], ['1', 'Yes, Version 1'], ['2', 'Yes, Version 2'], ['3', 'Yes, Version 3']])}
    <div class="card"><p class="q">Anything else you noticed? (optional)</p><textarea name="C7" maxlength="1500"></textarea></div>`,
  async (d) => {
    s.answers.compare = d;
    await send(s, 'compare', { order: s.order, answers: d });
    go('done');
  }, 'Finish');
}

function done() {
  if (s.stage === 'ineligible') return check();
  const unsent = (s.sent || []).filter((x) => !x.ok).length;
  app.innerHTML = `<h1>Thank you!</h1>
    <p>Your participant code:</p><p class="id">${esc(s.id)}</p>
    <p class="muted">Keep this code if you may want your answers deleted later.</p>
    ${unsent ? `<p class="muted">${unsent} part(s) could not be sent from this phone (no connection or no store set up); they are kept on this phone.</p>` : ''}`;
}

// Load the three trackers' files while the participant reads (so no version is recognisable by a longer wait):
// the vocabulary into the Cache API the SLAM worker reads it from (slam.worker.js cacheVocab), the rest into the
// HTTP cache.
async function preload() {
  try {
    const c = await caches.open('webvio-frames-v1');
    if (!(await c.match(VOCAB))) { const r = await fetch(VOCAB); if (r.ok) await c.put(VOCAB, r); }
  } catch { /* no Cache API: the WebVIO version downloads it when it starts */ }
  for (const u of ['../build/orbslam3.wasm', '../build/orbslam3.js', '../assets/tree.glb', './vendor/alva_ar.js']) {
    fetch(new URL(u, location.href)).catch(() => {});
  }
}

if (s.stage !== 'consent' && s.stage !== 'done') preload();
render();
