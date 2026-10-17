// WebVIO recorder: records a phone sequence for repeatable tests. Camera frames (grayscale JPEG, record.worker.js),
// DeviceMotion gyro + accelerometer (the same source and axes the AR page uses), the Generic Sensor API gyroscope and
// accelerometer when available (higher rate, separate file), and taps on the preview (image pixels). Everything is on one
// clock (page timeOrigin + ms, as ar.js). Stored by the dev server as an EuRoC-layout sequence in
// datasets/phone/<name>/ (mav0/cam0, mav0/imu0, touches.csv, meta.json, settings.yaml), which the AR page replays
// (ar.html?seq=phone/<name>, every preset, also on the phone) and the native / WASM runners read (--dataset euroc).
import { cameraSettings, intrinsics, loadCalibration, phoneImuSettings } from './slamclient.js';

const $ = (id) => document.getElementById(id);
const now = () => performance.timeOrigin + performance.now();
// ms on the page clock -> integer ns, exactly (epoch ms x 1e6 does not fit a double's 53 bits)
const ns = (ms) => { const w = Math.floor(ms); return BigInt(w) * 1000000n + BigInt(Math.round((ms - w) * 1e6)); };

let stream = null, track = null, worker = null, recording = false, t0 = 0, W = 0, H = 0, name = '';
let motion = [], gsGyro = [], gsAcc = [], touches = [], gsSensors = [];
let motionN = 0, motionT0 = 0, motionT1 = 0;

function stamp() {
  const d = new Date(), p = (v) => String(v).padStart(2, '0');
  return `rec_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
$('name').value = stamp();

function status(s, cls = '') { $('status').textContent = s; $('status').className = cls; }

async function put(file, body) {
  const r = await fetch(`/api/record?name=${encodeURIComponent(name)}&file=${encodeURIComponent(file)}`, { method: 'POST', body });
  if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
}

function onMotion(e) {
  if (!recording) return;
  const r = e.rotationRate, a = e.accelerationIncludingGravity;
  if (!r || r.alpha == null || !a || a.x == null) return;
  const d = Math.PI / 180;  // device axes = IMU body; rotationRate alpha/beta/gamma = z/x/y in deg/s (as ar.js)
  motion.push([performance.timeOrigin + e.timeStamp, r.beta * d, r.gamma * d, r.alpha * d, a.x, a.y, a.z]);
  if (!motionN) motionT0 = e.timeStamp;
  motionN++; motionT1 = e.timeStamp;
}

function startGenericSensors() {  // optional: Gyroscope / Accelerometer at up to 200 Hz (Chrome caps the rate itself)
  const mk = (Cls, out) => {
    try {
      const s = new Cls({ frequency: 200 });
      s.addEventListener('reading', () => { if (recording) out.push([performance.timeOrigin + s.timestamp, s.x, s.y, s.z]); });
      s.addEventListener('error', (ev) => console.warn(Cls.name, ev.error && ev.error.message));
      s.start();
      gsSensors.push(s);
    } catch (err) { console.warn(Cls.name, err.message); }
  };
  if ('Gyroscope' in self) mk(self.Gyroscope, gsGyro);
  if ('Accelerometer' in self) mk(self.Accelerometer, gsAcc);
}

// taps on the preview -> image pixels (the video is shown with object-fit: contain)
$('preview').addEventListener('pointerdown', (e) => {
  if (!recording || !W) return;
  const r = $('preview').getBoundingClientRect();
  const s = Math.min(r.width / W, r.height / H), ox = r.left + (r.width - W * s) / 2, oy = r.top + (r.height - H * s) / 2;
  const u = (e.clientX - ox) / s, v = (e.clientY - oy) / s;
  if (u < 0 || v < 0 || u >= W || v >= H) return;
  touches.push([performance.timeOrigin + e.timeStamp, u, v]);
  const dot = $('tapdot');
  dot.style.left = `${e.clientX - r.left}px`; dot.style.top = `${e.clientY - r.top}px`;
  dot.hidden = false; dot.classList.remove('pulse'); void dot.offsetWidth; dot.classList.add('pulse');
  $('taps').textContent = touches.length;
});

async function openCamera() {
  const [rw, rh] = $('res').value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: rw }, height: { ideal: rh }, frameRate: { ideal: 30 } }, audio: false });
  track = stream.getVideoTracks()[0];
  $('preview').srcObject = stream;
  await $('preview').play();
  W = $('preview').videoWidth; H = $('preview').videoHeight;
  $('size').textContent = `${W}×${H}`;
}

async function start() {
  name = $('name').value.trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) { status('Name: letters, digits, _ and - only (up to 64).', 'err'); return; }
  $('go').disabled = true;
  try {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function')
      await DeviceMotionEvent.requestPermission().catch(() => {});
    addEventListener('devicemotion', onMotion);
    if (!stream) await openCamera();
    if (!('MediaStreamTrackProcessor' in self)) throw new Error('this browser has no MediaStreamTrackProcessor (use Chrome)');
    motion = []; gsGyro = []; gsAcc = []; touches = []; motionN = 0;
    startGenericSensors();
    const proc = new MediaStreamTrackProcessor({ track: track.clone() });
    worker = new Worker('record.worker.js');
    worker.onmessage = onWorker;
    recording = true; t0 = now();
    worker.postMessage({ type: 'start', name, readable: proc.readable, quality: Number($('quality').value) }, [proc.readable]);
    $('stop').disabled = false; $('rec').hidden = false;
    status('Recording. Tap the picture to mark a point (for example the floor where a tree should go).');
  } catch (err) {
    status(`Could not start: ${err.message}`, 'err');
    $('go').disabled = false;
  }
}

let last = {};
function onWorker(e) {
  const m = e.data;
  if (m.type === 'stats') {
    last = m;
    const secs = recording ? (now() - t0) / 1000 : 0;
    const hz = motionN > 1 ? (motionN - 1) * 1000 / (motionT1 - motionT0) : 0;
    $('time').textContent = secs.toFixed(1);
    $('frames').textContent = m.frames;
    $('fps').textContent = secs > 0.5 ? (m.frames / secs).toFixed(1) : '–';
    $('dropped').textContent = m.dropped;
    $('gyro').textContent = hz ? hz.toFixed(0) : '–';
    $('gs').textContent = gsGyro.length && secs > 0.5 ? (gsGyro.length / secs).toFixed(0) : '–';
    $('queue').textContent = m.queued;
    $('mb').textContent = (m.bytes / 1048576).toFixed(1);
  } else if (m.type === 'error') {
    console.warn(m.text);
    $('errors').textContent = m.text;
  } else if (m.type === 'done') finish(m);
}

function stop() {
  if (!recording) return;
  recording = false;
  $('stop').disabled = true; $('rec').hidden = true;
  removeEventListener('devicemotion', onMotion);
  for (const s of gsSensors) try { s.stop(); } catch {}
  gsSensors = [];
  status('Uploading the last frames…');
  worker.postMessage({ type: 'stop' });
}

async function finish(m) {
  try {
    // camera timestamps -> page clock: the smallest (receipt - capture) is the capture -> receipt latency floor (as
    // slam.worker.js); one offset for the whole recording keeps the frame intervals exactly the camera's
    const frames = m.frames.sort((a, b) => a[0] - b[0]);
    if (!frames.length) throw new Error('no frames were stored');
    const off = Math.min(...frames.map(([, ts, recv]) => recv - ts / 1000));
    const cam = ['#timestamp [ns],filename', ...frames.map(([i, ts]) => `${ns(ts / 1000 + off)},frame_${i}.jpg`)];
    const imu = ['#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]',
      ...motion.sort((a, b) => a[0] - b[0]).filter((r, k, arr) => !k || r[0] > arr[k - 1][0]).map((r) => [ns(r[0]), ...r.slice(1).map((v) => v.toFixed(6))].join(','))];
    const tap = ['#timestamp [ns],u [px],v [px]', ...touches.map(([t, u, v]) => `${ns(t)},${u.toFixed(1)},${v.toFixed(1)}`)];
    const gs = (rows) => ['#timestamp [ns],x,y,z', ...rows.map(([t, x, y, z]) => `${ns(t)},${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`)].join('\n') + '\n';
    const secs = (frames[frames.length - 1][2] - frames[0][2]) / 1000;
    const hz = motionN > 1 ? (motionN - 1) * 1000 / (motionT1 - motionT0) : 60;
    const hfov = Number($('fov').value), calib = $('usecal').checked ? loadCalibration(W, H) : null;
    const fps = (frames.length - 1) / Math.max(secs, 1e-3);
    const meta = {
      name, notes: $('notes').value, recorded: new Date(t0).toISOString(), userAgent: navigator.userAgent, cores: navigator.hardwareConcurrency,
      image: { width: W, height: H }, fps: +fps.toFixed(2), frames: frames.length, droppedByRecorder: m.dropped, jpegQuality: Number($('quality').value),
      intrinsics: intrinsics(W, H, hfov, calib), hfovAssumed: calib ? null : hfov, calibrated: !!calib,
      imu: { source: 'DeviceMotion (rotationRate, accelerationIncludingGravity; device axes)', samples: motion.length, hz: +hz.toFixed(1),
             genericSensor: { gyroscope: gsGyro.length, accelerometer: gsAcc.length } },
      cameraClock: { offsetMs: +off.toFixed(3), note: 'frame t = VideoFrame.timestamp/1000 + offset (min receipt - capture), page clock' },
      cameraToImu: 'IMU.T_b_c1 = diag(1, -1, -1) (rear camera, portrait), as slamclient.js phoneImuSettings',
      taps: touches.length, durationS: +secs.toFixed(2),
    };
    const settings = cameraSettings(W, H, hfov, fps, 1000, calib) + phoneImuSettings(hz);
    status('Saving the sequence files…');
    await put('data.csv', cam.join('\n') + '\n');
    await put('imu.csv', imu.join('\n') + '\n');
    await put('touches.csv', tap.join('\n') + '\n');
    if (gsGyro.length) await put('gyro_gs.csv', gs(gsGyro));
    if (gsAcc.length) await put('accel_gs.csv', gs(gsAcc));
    await put('settings.yaml', settings);
    await put('meta.json', JSON.stringify(meta, null, 2));
    status(`Saved "${name}": ${frames.length} frames (${fps.toFixed(1)} fps), ${motion.length} IMU samples (${hz.toFixed(0)} Hz), ${touches.length} taps, ${secs.toFixed(1)} s${m.failed ? `, ${m.failed} frames failed to upload` : ''}.`, 'ok');
    $('replay').href = `ar.html?seq=phone/${encodeURIComponent(name)}`;
    $('replay').hidden = false;
    listRecordings();
  } catch (err) {
    status(`Saving failed: ${err.message}`, 'err');
  }
  worker.terminate();
  $('name').value = stamp();
  $('go').disabled = false;
}

async function listRecordings() {
  try {
    const seqs = (await (await fetch('/api/sequences')).json()).filter((s) => s.dataset === 'phone');
    $('list').innerHTML = seqs.length ? seqs.map((s) => `<li><a href="ar.html?seq=phone/${encodeURIComponent(s.name)}">${s.name}</a></li>`).join('')
                                      : '<li class="dim">No recordings yet.</li>';
  } catch { $('list').innerHTML = '<li class="dim">The recording list needs the dev server (web/serve.py).</li>'; }
}

$('go').addEventListener('click', start);
$('stop').addEventListener('click', stop);
listRecordings();
