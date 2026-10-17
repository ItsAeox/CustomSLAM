// Algorithm / settings variants selectable in the viewer and the AR page (and via URL parameters, so a test can be
// reproduced exactly). They are applied on top of the ORB-SLAM3 settings text; the Web.* keys are read by
// slam/orbslam3/include/WebConfig.h. Defaults = the hybrid tracker with otherwise stock ORB-SLAM3 settings.

export const OPTIONS = [
  { key: 'tracker', label: 'Tracker', values: [['klt', 'Hybrid ORB + KLT'], ['stock', 'ORB on every frame (baseline)']], def: 'klt' },
  { key: 'desc', label: 'Descriptor', values: [['orb', 'ORB (rotated BRIEF)'], ['teblid', 'TEBLID-256 (own vocabulary)']], def: 'orb' },
  { key: 'features', label: 'ORB features', values: [['0', 'settings default'], ['500', '500'], ['600', '600'], ['800', '800'], ['1000', '1000'], ['1500', '1500']], def: '0' },
  { key: 'levels', label: 'Pyramid levels', values: [['0', 'settings default (8)'], ['4', '4'], ['5', '5'], ['6', '6'], ['8', '8']], def: '0' },
  { key: 'reloc', label: 'Relocalise before new map', values: [['3', '3 s (stock)'], ['10', '10 s'], ['30', '30 s']], def: '3' },
  { key: 'motion', label: 'Motion model', values: [['stock', 'stock (constant velocity per frame)'], ['time', 'time-scaled (dropped frames)']], def: 'stock' },
  { key: 'imu', label: 'IMU (mono-inertial, experimental)', page: 'ar', values: [['off', 'off (visual only)'], ['on', 'on (DeviceMotion)']], def: 'off' },
  { key: 'imudt', label: 'IMU time offset (ms, measured with "Record")', page: 'ar',
    values: [['0', '0 (none)'], ['-40', '−40'], ['-20', '−20'], ['20', '+20'], ['40', '+40'], ['60', '+60'], ['80', '+80'],
             ['100', '+100'], ['120', '+120']], def: '0' },
  { key: 'gyro', label: 'Gyro rotation prior (visual-only tracking)', page: 'ar',
    values: [['off', 'off'], ['on', 'on (predict rotation from the gyro)']], def: 'off' },
  { key: 'rec', label: 'Record gyro + poses (diagnostic)', page: 'ar',
    values: [['off', 'off'], ['on', 'on (log for the camera–IMU time offset)']], def: 'off' },
  { key: 'anchor', label: 'Tracking lost with a tree placed', page: 'ar',
    values: [['keep', 'keep the map, relocalise only'], ['stock', 'stock (new map after the timeout)']], def: 'keep' },
  { key: 'map', label: 'Local mapping', values: [['stock', 'stock'], ['light', 'light (local BA 20/20, 15 neighbours)']], def: 'stock' },
  { key: 'kltmax', label: 'KLT frames between ORB', values: [['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], def: '2', when: (o) => o.tracker === 'klt' },
  { key: 'kltwin', label: 'KLT window', values: [['11', '11 px'], ['15', '15 px'], ['21', '21 px']], def: '15', when: (o) => o.tracker === 'klt' },
  { key: 'kltlev', label: 'KLT pyramid levels', values: [['1', '1'], ['2', '2'], ['3', '3']], def: '2', when: (o) => o.tracker === 'klt' },
  { key: 'kltflow', label: 'KLT motion gate (max predicted flow)', values: [['0', 'off'], ['10', '10 px'], ['20', '20 px'], ['30', '30 px']], def: '0', when: (o) => o.tracker === 'klt' },
  { key: 'kltfrac', label: 'KLT quality gate (inliers vs last ORB frame)', values: [['0', 'off'], ['0.5', '50%'], ['0.7', '70%']], def: '0', when: (o) => o.tracker === 'klt' },
  { key: 'kltfb', label: 'KLT forward-backward check', values: [['1', 'on'], ['0', 'off']], def: '1', when: (o) => o.tracker === 'klt' },
  // round 2 (2026-10-17; claude/RESEARCH.md "Algorithm tracker"): defaults = previous behaviour
  { key: 'kltnorm', label: 'KLT exposure normalisation', values: [['off', 'off'], ['on', 'on (cancels exposure changes)']], def: 'off', when: (o) => o.tracker === 'klt' },
  { key: 'pose', label: 'Pose solver', values: [['g2o', 'g2o (original)'], ['fast', 'fast (same result, no g2o graph)']], def: 'g2o' },
  { key: 'orbpar', label: 'ORB extraction', values: [['serial', 'one thread (original)'], ['parallel', 'pyramid levels in parallel']], def: 'serial' },
  { key: 'recover', label: 'Recovery when lost', values: [['bow', 'bag-of-words relocalisation (original)'], ['coast', 'local-map search first, then bag of words']], def: 'bow' },
  { key: 'init', label: 'Map start', values: [['stock', 'original thresholds'], ['quick', 'quicker (60 matches, 0.5°, 40 points)'], ['gyro', 'gyro two-view (+ gyro prior)']], def: 'stock' },
  { key: 'kf', label: 'Keyframes', values: [['stock', 'original'], ['fewer', 'fewer (ratio 0.8)']], def: 'stock' },
  { key: 'hold', label: 'Tree during brief tracking losses', page: 'ar', values: [['hide', 'hidden (original)'], ['gyro', 'held by the gyro for up to 0.4 s']], def: 'hide' },
];

// Named presets (AR page): each sets every option; keys it does not list take the page default. Phone test plans
// (claude/PHONE.md) refer to them by name, and the session log records which one was active.
export const PRESETS = [
  { key: 'stock', label: 'Baseline (ORB every frame)', set: { tracker: 'stock', reloc: '3', anchor: 'stock' },
    note: 'ORB features on every frame and the original ORB-SLAM3 behaviour (a new map after 3 s lost): the reference the other presets are compared with.' },
  { key: 'hybrid', label: 'Hybrid ORB + KLT (default)', set: {},
    note: 'Cheap KLT optical-flow frames between ORB frames (about 2× faster tracking); keeps the map when lost.' },
  { key: 'fast', label: 'Hybrid, fastest', set: { kltmax: '4', motion: 'time', map: 'light' },
    note: 'Up to 4 KLT frames, motion prediction scaled to the frame interval, lighter local mapping (best phone fps so far).' },
  { key: 'light', label: 'Light ORB', set: { tracker: 'stock', features: '600', levels: '6' },
    note: 'ORB on every frame with 600 features and 6 pyramid levels (23% less tracking time, same accuracy on EuRoC).' },
  { key: 'imu', label: 'IMU (visual-inertial)', set: { imu: 'on', tracker: 'stock' },
    note: 'Visual-inertial mode with the phone IMU: real-world scale and gravity. Experimental; ORB on every frame. Needs a calibrated camera and the IMU time offset.' },
  { key: 'imufast', label: 'IMU, tuned for speed', set: { imu: 'on', tracker: 'stock', motion: 'time', features: '600', levels: '6', map: 'light' },
    note: 'IMU plus time-scaled motion, light ORB and lighter local mapping (tests whether speed is what breaks IMU mode).' },
];

// Sets every option of `page` from a preset (unlisted keys: page default, then option default)
export function applyPreset(o, preset, pageDefaults = {}, page = '') {
  for (const d of OPTIONS) if (!d.page || d.page === page) o[d.key] = preset.set[d.key] ?? pageDefaults[d.key] ?? d.def;
  return o;
}

// Key of the preset the options equal (options hidden by `when`, e.g. KLT knobs with stock tracking, are ignored), or 'custom'
export function matchPreset(o, pageDefaults = {}, page = '') {
  const p = PRESETS.find((pr) => OPTIONS.every((d) => (d.page && d.page !== page) || !(d.key in o) || (d.when && !d.when(o)) ||
                                                      o[d.key] === (pr.set[d.key] ?? pageDefaults[d.key] ?? d.def)));
  return p ? p.key : 'custom';
}

// pageDefaults: per-page defaults (e.g. the AR page relocalises longer); URL parameters override them.
// page: options marked with a page (e.g. 'ar') exist only there; the returned object holds the page's options.
export function optionsFromParams(params, pageDefaults = {}, page = '') {
  const o = {};
  for (const d of OPTIONS) if (!d.page || d.page === page) o[d.key] = params.get(d.key) ?? pageDefaults[d.key] ?? d.def;
  return o;
}

// URL parameters for the options that differ from the page defaults (the same as optionsFromParams got)
export function optionsToParams(o, params = new URLSearchParams(), pageDefaults = {}) {
  for (const d of OPTIONS) {
    if (!(d.key in o)) continue;
    if (o[d.key] !== (pageDefaults[d.key] ?? d.def)) params.set(d.key, o[d.key]); else params.delete(d.key);
  }
  return params;
}

export function describeOptions(o) {
  return OPTIONS.filter((d) => d.key in o && (!d.when || d.when(o))).map((d) => `${d.key}=${o[d.key]}`).join(' ');
}

// ORB-SLAM3 settings text + options -> settings text
export function applyOptions(yaml, o) {
  let y = yaml;
  if (Number(o.features) > 0) y = y.replace(/(ORBextractor\.nFeatures:\s*)\d+/, `$1${Number(o.features)}`);
  if (Number(o.levels) > 0) y = y.replace(/(ORBextractor\.nLevels:\s*)\d+/, `$1${Number(o.levels)}`);
  const extra = [];
  if (o.tracker === 'klt') {
    extra.push('Web.KLT.Enable: 1', `Web.KLT.MaxConsecutive: ${Number(o.kltmax)}`, `Web.KLT.WinSize: ${Number(o.kltwin)}`,
               `Web.KLT.Levels: ${Number(o.kltlev)}`, `Web.KLT.FBThreshold: ${o.kltfb === '0' ? 0 : 1.0}`,
               `Web.KLT.MaxFlow: ${Number(o.kltflow)}`, `Web.KLT.MinInlierFrac: ${Number(o.kltfrac)}`);
  }
  if (Number(o.reloc) !== 3) extra.push(`Web.Track.RelocTimeout: ${Number(o.reloc)}`);
  if (o.desc === 'teblid') extra.push('Web.Descriptor: 1');  // the page also loads the TEBLID vocabulary (vocabUrl)
  if (o.motion === 'time') extra.push('Web.Track.TimeScaledMotion: 1');  // live camera: prediction scaled to the frame interval
  if (o.map === 'light') extra.push('Web.LBA.MaxOptKFs: 20', 'Web.LBA.MaxFixedKFs: 20', 'Web.Mapping.Neighbors: 15');
  if ((o.gyro === 'on' || o.init === 'gyro') && o.imu !== 'on') extra.push('Web.Gyro.Prior: 1');  // + IMU.T_b_c1 (ar.js adds the phone IMU block)
  if (o.tracker === 'klt' && o.kltnorm === 'on') extra.push('Web.KLT.Normalize: 1');
  if (o.pose === 'fast') extra.push('Web.Track.FastPose: 1');
  if (o.orbpar === 'parallel') extra.push('Web.ORB.Parallel: 1');
  if (o.recover === 'coast') extra.push('Web.Track.Coast: 2');
  if (o.init === 'quick') extra.push('Web.Init.MinMatches: 60', 'Web.Init.MinParallax: 0.5', 'Web.Init.MinTriangulated: 40');
  if (o.init === 'gyro' && o.imu !== 'on') extra.push('Web.Init.Gyro: 1');
  if (o.kf === 'fewer') extra.push('Web.KF.RefRatio: 0.8');
  return extra.length ? `${y.replace(/\s*$/, '\n')}\n# web options (web/options.js)\n${extra.join('\n')}\n` : y;
}

// Renders <label><select> controls into `container`; calls onChange(options) after every change.
export function buildOptionsUI(container, o, onChange = () => {}) {
  container.textContent = '';
  const rows = [];
  for (const d of OPTIONS) {
    if (!(d.key in o)) continue;
    const label = document.createElement('label');
    label.className = 'opt';
    label.append(`${d.label} `);
    const sel = document.createElement('select');
    for (const [v, text] of d.values) sel.append(new Option(text, v));
    sel.value = o[d.key];
    sel.onchange = () => { o[d.key] = sel.value; refresh(); onChange(o); };
    label.append(sel);
    container.append(label);
    rows.push([d, label]);
  }
  const refresh = () => { for (const [d, label] of rows) label.hidden = !!d.when && !d.when(o); };
  refresh();
  return o;
}
