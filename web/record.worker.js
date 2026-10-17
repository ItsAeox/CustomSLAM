// WebVIO recorder worker: reads the camera's VideoFrames (MediaStreamTrackProcessor stream handed over by record.js),
// keeps each frame's capture timestamp and receipt time, encodes it as a grayscale JPEG (the SLAM uses the luma image)
// and uploads it to the dev server (web/serve.py POST /api/record) while recording, so the phone holds only a short queue.
// Frames are named by index; the final timestamps (IMU clock) are written by record.js when the recording stops.

let reader = null, name = '', quality = 0.92, stop = false;
const frames = [];            // [index, capture timestamp (us), receipt time (ms, timeOrigin + now)]
let encoding = 0, dropped = 0, queued = 0, uploaded = 0, bytes = 0, failed = 0;
const uploads = [];           // pending { file, blob, i }
const ok = new Set();         // frame indices stored on the server
let active = 0;
const MAX_ENCODING = 2, MAX_UPLOADS = 4, MAX_QUEUE = 240;  // ~8 s of frames waiting for the network

function pump() {
  while (active < MAX_UPLOADS && uploads.length) {
    const u = uploads.shift();
    active++;
    const put = (tries) => fetch(`/api/record?name=${encodeURIComponent(name)}&file=${encodeURIComponent(u.file)}`, { method: 'POST', body: u.blob })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); uploaded++; bytes += u.blob.size; ok.add(u.i); })
      .catch((e) => (tries > 0 ? new Promise((res) => setTimeout(res, 500)).then(() => put(tries - 1)) : (failed++, postMessage({ type: 'error', text: `${u.file}: ${e.message}` }))));
    put(3).finally(() => { active--; queued--; pump(); });
  }
}

async function encode(frame, index) {
  const w = frame.displayWidth, h = frame.displayHeight;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.filter = 'grayscale(1)';
  ctx.drawImage(frame, 0, 0, w, h);
  frame.close();
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  uploads.push({ file: `frame_${index}.jpg`, blob, i: index });
  queued++;
  pump();
  return { w, h };
}

let size = null;
async function loop() {
  let index = 0;
  for (;;) {
    let r;
    try { r = await reader.read(); } catch { break; }
    if (r.done) break;
    const f = r.value;
    if (stop) { f.close(); continue; }
    const recv = performance.timeOrigin + performance.now();
    if (encoding >= MAX_ENCODING || queued >= MAX_QUEUE) { f.close(); dropped++; continue; }  // keep up: drop, count
    const i = index++;
    frames.push([i, f.timestamp, recv]);
    encoding++;
    encode(f, i).then((s) => { size = size || s; }).catch((e) => postMessage({ type: 'error', text: `encode: ${e.message}` }))
      .finally(() => { encoding--; });
  }
}

setInterval(() => postMessage({ type: 'stats', frames: frames.length, dropped, queued, uploaded, bytes, failed, size }), 500);

onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'start') {
    name = m.name; quality = m.quality || quality;
    reader = m.readable.getReader();
    loop();
  } else if (m.type === 'stop') {
    stop = true;
    try { await reader.cancel(); } catch {}
    while (encoding > 0 || queued > 0) await new Promise((res) => setTimeout(res, 100));  // every frame uploaded
    postMessage({ type: 'done', frames: frames.filter((f) => ok.has(f[0])), dropped, uploaded, bytes, failed, size });
  }
};
