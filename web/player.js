// Dataset player: runs ORB-SLAM3 (WebAssembly, in a Worker) over a dataset sequence in the browser,
// frame by frame without dropping frames, and uploads the results for offline evaluation:
//   <name>_live.txt    per-frame poses as returned live by TrackMonocular (what an AR app would use), TUM format
//   <name>_final.txt   ORB-SLAM3's final optimised trajectory (same as the native driver), TUM format
//   <name>_timing.csv  per-frame track / decode times (ms) and tracking state
//
// URL: player.html?seq=mocap-desk2&settings=tumvie_mocap-desk2.yaml[&tag=run1][&max=N]

const params = new URLSearchParams(location.search);
const seqName = params.get('seq') || 'mocap-desk2';
const settingsFile = params.get('settings') || `tumvie_${seqName}.yaml`;
const tag = params.get('tag') || 'run1';
const maxFrames = Number(params.get('max') || 0);
const seed = params.has('seed') ? Number(params.get('seed')) : -1;  // -1: ORB-SLAM3 default
// realtime=1: pace frames at the dataset rate like ORB-SLAM3's examples (sleep for the rest of each frame period).
const realtime = params.get('realtime') === '1';
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');

const STATE = { '-1': 'not ready', 0: 'no images', 1: 'not initialised', 2: 'OK', 3: 'recently lost', 4: 'lost', 5: 'OK (KLT)' };

function log(s) { logEl.textContent += s + '\n'; logEl.scrollTop = logEl.scrollHeight; }
function setStatus(s) { statusEl.textContent = s; }

// column-major 4x4 -> TUM line "t tx ty tz qx qy qz qw"
function tumLine(t, m) {
  const r00 = m[0], r10 = m[1], r20 = m[2], r01 = m[4], r11 = m[5], r21 = m[6], r02 = m[8], r12 = m[9], r22 = m[10];
  const tr = r00 + r11 + r22;
  let x, y, z, w;
  if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; w = 0.25 * S; x = (r21 - r12) / S; y = (r02 - r20) / S; z = (r10 - r01) / S; }
  else if (r00 > r11 && r00 > r22) { const S = Math.sqrt(1 + r00 - r11 - r22) * 2; w = (r21 - r12) / S; x = 0.25 * S; y = (r01 + r10) / S; z = (r02 + r20) / S; }
  else if (r11 > r22) { const S = Math.sqrt(1 + r11 - r00 - r22) * 2; w = (r02 - r20) / S; x = (r01 + r10) / S; y = 0.25 * S; z = (r12 + r21) / S; }
  else { const S = Math.sqrt(1 + r22 - r00 - r11) * 2; w = (r10 - r01) / S; x = (r02 + r20) / S; y = (r12 + r21) / S; z = 0.25 * S; }
  return `${t.toFixed(9)} ${m[12]} ${m[13]} ${m[14]} ${x} ${y} ${z} ${w}`;
}

// ORB-SLAM3 EuRoC-format trajectory (ns timestamps of the rebased clock) -> TUM with absolute seconds
function euRocToTum(text, tBase) {
  const out = ['# t tx ty tz qx qy qz qw'];
  for (const line of text.split('\n')) {
    const v = line.trim().split(/\s+/).map(Number);
    if (v.length < 8 || v.some((x) => !Number.isFinite(x))) continue;
    out.push(`${(v[0] * 1e-9 + tBase).toFixed(9)} ${v.slice(1, 8).join(' ')}`);
  }
  return out.join('\n') + '\n';
}

async function upload(name, text) {
  const r = await fetch(`/api/result?name=${encodeURIComponent(name)}`, { method: 'POST', body: text });
  if (!r.ok) throw new Error(`upload ${name}: ${r.status}`);
}

async function main() {
  if (!crossOriginIsolated) {
    setStatus('Not cross-origin isolated: SharedArrayBuffer (WASM threads) unavailable. Use web/serve.py.');
    return;
  }
  const manifest = await (await fetch(`/api/sequence?dataset=tumvie&name=${encodeURIComponent(seqName)}`)).json();
  const frames = maxFrames > 0 ? manifest.frames.slice(0, maxFrames) : manifest.frames;
  log(`${seqName}: ${frames.length} frames (${manifest.missing} missing), settings ${settingsFile}, threads: ${navigator.hardwareConcurrency}`);

  const worker = new Worker('./slam.worker.js', { type: 'module' });
  let pending = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') { if (!/^\s*$/.test(m.text)) log(m.text); return; }
    if (m.type === 'error') { log('ERROR ' + m.text); setStatus('error (see log)'); return; }
    if (pending) { const p = pending; pending = null; p(m); }
  };
  const request = (msg) => new Promise((resolve) => { pending = resolve; worker.postMessage(msg); });

  setStatus('loading WASM + vocabulary...');
  const ready = await request({ type: 'init', vocabUrl: '/vocab/ORBvoc.bin', settingsUrl: `/settings/${settingsFile}`,
                                clahe: manifest.clahe, seed });
  log(`init: ${ready.ms.toFixed(0)} ms (vocabulary ${(ready.vocabBytes / 1e6).toFixed(1)} MB), ` +
      `wasm heap ${(ready.heap / 1048576).toFixed(0)} MiB, seed ${seed}, realtime ${realtime}`);
  let peakHeap = ready.heap;

  const live = ['# t tx ty tz qx qy qz qw  (live TrackMonocular poses, Twc)'];
  const timing = ['idx,t,state,track_ms,decode_ms'];
  const t0 = performance.now();
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const tf = performance.now();
    const r = await request({ type: 'frame', idx: f.idx, t: f.t, url: f.url });
    if (realtime && i + 1 < frames.length) {
      const remain = (frames[i + 1].t - f.t) * 1000 - (performance.now() - tf);
      if (remain > 0) await new Promise((res) => setTimeout(res, remain));
    }
    timing.push(`${f.idx},${f.t.toFixed(6)},${r.state},${r.trackMs.toFixed(3)},${r.decodeMs.toFixed(3)}`);
    peakHeap = Math.max(peakHeap, r.heap);
    if (r.state === 2 || r.state === 5) live.push(tumLine(f.t + manifest.tBase, r.twc));
    if (i % 20 === 0 || i === frames.length - 1) {
      setStatus(`frame ${i + 1}/${frames.length}  state: ${STATE[r.state]}  track ${r.trackMs.toFixed(1)} ms`);
    }
  }
  const wallS = (performance.now() - t0) / 1000;
  setStatus('finishing (final optimisation + save)...');
  const fin = await request({ type: 'finish' });

  const base = `${seqName}_${tag}`;
  await upload(`${base}_live.txt`, live.join('\n') + '\n');
  await upload(`${base}_final.txt`, euRocToTum(fin.frames, manifest.tBase));
  await upload(`${base}_timing.csv`, timing.join('\n') + '\n');
  const ms = timing.slice(1).map((l) => Number(l.split(',')[3])).sort((a, b) => a - b);
  const med = ms[Math.floor(ms.length / 2)];
  // Two different speeds: TrackMonocular time inside the worker, and end-to-end throughput of this page
  // (fetch + decode + messaging + tracking, frames processed sequentially without dropping).
  const msg = `done: ${frames.length} frames in ${wallS.toFixed(1)} s | end-to-end ${(frames.length / wallS).toFixed(1)} fps | ` +
              `median TrackMonocular ${med.toFixed(1)} ms (${(1000 / med).toFixed(1)} fps) | ` +
              `peak wasm heap ${(peakHeap / 1048576).toFixed(0)} MiB | uploaded ${base}_{live,final,timing}`;
  log(msg);
  setStatus(msg);
  document.title = 'DONE ' + seqName;
}

main().catch((err) => { log('ERROR ' + (err.stack || err)); setStatus('error (see log)'); });
