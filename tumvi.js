// tumvi.js - loader for your loop-floor0 style dataset
// Folder layout expected:
//   left_images/{00000.jpg,..., image_timestamps_left.txt, image_exposures_left.txt}
//   right_images/{... similar ...}
//   camera-calibration.json
//   imu_data.txt
//   mocap_data.txt

function normPath(s) { return String(s || '').replace(/\\/g, '/'); }

function findFile(fileList, suffix) {
  const suf = normPath(suffix);
  for (const f of fileList) {
    const p = normPath(f.webkitRelativePath || f.name);
    if (p === suf || p.endsWith('/' + suf) || p.endsWith(suf)) return f;
  }
  return null;
}

function findFilesUnder(fileList, dirPrefix, exts) {
  const pref = normPath(dirPrefix).replace(/\/+$/,'') + '/';
  const out = [];
  for (const f of fileList) {
    const p = normPath(f.webkitRelativePath || f.name);
    if (!p.includes(pref)) continue;
    const lower = p.toLowerCase();
    if (exts.some(e => lower.endsWith(e))) out.push(f);
  }
  out.sort((a,b)=> normPath(a.webkitRelativePath||a.name).localeCompare(normPath(b.webkitRelativePath||b.name), undefined, { numeric:true }));
  return out;
}

async function readText(file) { return await file.text(); }

function parseUsFloatLines(text) {
  // Lines like: 1.226649075999999885e+06
  // Ignore comments starting with '#'
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const out = [];
  for (const s of lines) {
    if (s.startsWith('#')) continue;
    const v = Number(s);
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

function parseImu(text) {
  // Header says:
  // # calibrated imu data gx(rad/s), gy(rad/s), gz(rad/s), ax(m/s^2), ay(m/s^2), az(m/s^2)
  // Then each line:
  // t_us gx gy gz ax ay az [temp?]
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const out = [];
  for (const s of lines) {
    if (s.startsWith('#')) continue;
    const v = s.split(/\s+/).map(Number);
    if (v.length < 7 || v.some(x => !Number.isFinite(x))) continue;
    const t_us = v[0];
    const gx = v[1], gy = v[2], gz = v[3];
    const ax = v[4], ay = v[5], az = v[6];
    out.push({ t: t_us * 1e-6, ax, ay, az, wx: gx, wy: gy, wz: gz });
  }
  return out;
}

function parseMocap(text) {
  // # mocap poses: time(us) px py pz qx qy qz qw
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const out = [];
  for (const s of lines) {
    if (s.startsWith('#')) continue;
    const v = s.split(/\s+/).map(Number);
    if (v.length < 8 || v.some(x => !Number.isFinite(x))) continue;
    out.push({
      t: v[0] * 1e-6,
      px: v[1], py: v[2], pz: v[3],
      qx: v[4], qy: v[5], qz: v[6], qw: v[7],
    });
  }
  return out;
}

export async function loadTumviSequence(fileList, opts) {
  const leftDir  = opts.leftDir || 'left_images';
  const rightDir = opts.rightDir || 'right_images';

  const leftImgs = findFilesUnder(fileList, leftDir, ['.jpg', '.jpeg', '.png']);
  if (!leftImgs.length) throw new Error(`No images in ${leftDir}`);

  const tsLeftFile = findFile(fileList, `${leftDir}/image_timestamps_left.txt`);
  if (!tsLeftFile) throw new Error(`Missing ${leftDir}/image_timestamps_left.txt`);

  // Optional files
  const expLeftFile = findFile(fileList, `${leftDir}/image_exposures_left.txt`);
  const camCalibFile = findFile(fileList, `camera-calibration.json`);
  const imuFile = findFile(fileList, `imu_data.txt`);
  const mocapFile = findFile(fileList, `mocap_data.txt`);

  const Module = opts.Module;

  // Calibration (needed early for cam time offset)
  let calib = null;
  if (camCalibFile) {
    const raw = JSON.parse(await readText(camCalibFile));
    calib = raw?.value0 ?? raw; // <-- IMPORTANT: unwrap value0
  }

  let camOffset = 0.0;
  if (calib && Number.isFinite(calib.cam_time_offset_ns)) {
    camOffset = calib.cam_time_offset_ns * 1e-9; // seconds
  }

  const tsUs = parseUsFloatLines(await readText(tsLeftFile));
  let imageTS = tsUs.map(x => x * 1e-6 + camOffset);

  // Rebase to 0 like KITTI runner does
  const t0 = imageTS.length ? imageTS[0] : 0;
  imageTS = imageTS.map(t => t - t0);

  // Exposures (not used yet, but parsed for completeness)
  let exposuresUs = [];
  if (expLeftFile) exposuresUs = parseUsFloatLines(await readText(expLeftFile));

  // IMU stream
  let imuStream = [];
  if (imuFile) {
    const imuRaw = parseImu(await readText(imuFile));
    // Rebase to same t0 as images
    imuStream = imuRaw.map(s => ({ ...s, t: s.t - t0 }));
  }

  // Mocap (for future overlay/eval)
  let mocap = [];
  if (mocapFile) {
    const mocRaw = parseMocap(await readText(mocapFile));
    mocap = mocRaw.map(m => ({ ...m, t: m.t - t0 }));
  }

  // Right camera images (optional, not used yet)
  const rightImgs = findFilesUnder(fileList, rightDir, ['.jpg', '.jpeg', '.png']);

  const N = Math.min(leftImgs.length, imageTS.length || leftImgs.length);

  // Push IMU->Cam extrinsics into WASM (use camera 0 by default)
  // T_imu_cam entries are objects: {px,py,pz,qx,qy,qz,qw}
  if (calib && Array.isArray(calib.T_imu_cam) && calib.T_imu_cam.length >= 1) {
    const camIdx = (opts.camIdx ?? 0);
    const T = calib.T_imu_cam[camIdx];
    if (T && Module?.setImuToCamQuat) {
      Module.setImuToCamQuat(T.qx, T.qy, T.qz, T.qw, T.px, T.py, T.pz);
    }
  }

  return {
    kind: 'tumvi',
    leftImgs,
    rightImgs,
    imageTS,
    exposuresUs,
    imuStream,
    mocap,
    calib,
    N,
  };
}
