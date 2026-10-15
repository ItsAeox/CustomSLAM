// Algorithm / settings variants selectable in the viewer and the AR page (and via URL parameters, so a test can be
// reproduced exactly). They are applied on top of the ORB-SLAM3 settings text; the Web.* keys are read by
// slam/orbslam3/include/WebConfig.h. Defaults = the hybrid tracker with otherwise stock ORB-SLAM3 settings.

export const OPTIONS = [
  { key: 'tracker', label: 'Tracker', values: [['klt', 'Hybrid ORB + KLT'], ['stock', 'ORB-SLAM3 (ORB every frame)']], def: 'klt' },
  { key: 'features', label: 'ORB features', values: [['0', 'settings default'], ['500', '500'], ['600', '600'], ['800', '800'], ['1000', '1000'], ['1500', '1500']], def: '0' },
  { key: 'levels', label: 'Pyramid levels', values: [['0', 'settings default (8)'], ['4', '4'], ['5', '5'], ['6', '6'], ['8', '8']], def: '0' },
  { key: 'reloc', label: 'Relocalise before new map', values: [['3', '3 s (stock)'], ['10', '10 s'], ['30', '30 s']], def: '3' },
  { key: 'map', label: 'Local mapping', values: [['stock', 'stock'], ['light', 'light (local BA 20/20, 15 neighbours)']], def: 'stock' },
  { key: 'kltmax', label: 'KLT frames between ORB', values: [['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], def: '2', when: (o) => o.tracker === 'klt' },
  { key: 'kltwin', label: 'KLT window', values: [['11', '11 px'], ['15', '15 px'], ['21', '21 px']], def: '15', when: (o) => o.tracker === 'klt' },
  { key: 'kltlev', label: 'KLT pyramid levels', values: [['1', '1'], ['2', '2'], ['3', '3']], def: '2', when: (o) => o.tracker === 'klt' },
  { key: 'kltflow', label: 'KLT motion gate (max predicted flow)', values: [['0', 'off'], ['10', '10 px'], ['20', '20 px'], ['30', '30 px']], def: '0', when: (o) => o.tracker === 'klt' },
  { key: 'kltfrac', label: 'KLT quality gate (inliers vs last ORB frame)', values: [['0', 'off'], ['0.5', '50%'], ['0.7', '70%']], def: '0', when: (o) => o.tracker === 'klt' },
  { key: 'kltfb', label: 'KLT forward-backward check', values: [['1', 'on'], ['0', 'off']], def: '1', when: (o) => o.tracker === 'klt' },
];

// pageDefaults: per-page defaults (e.g. the AR page relocalises longer); URL parameters override them
export function optionsFromParams(params, pageDefaults = {}) {
  const o = {};
  for (const d of OPTIONS) o[d.key] = params.get(d.key) ?? pageDefaults[d.key] ?? d.def;
  return o;
}

export function optionsToParams(o, params = new URLSearchParams()) {
  for (const d of OPTIONS) if (o[d.key] !== d.def) params.set(d.key, o[d.key]); else params.delete(d.key);
  return params;
}

export function describeOptions(o) {
  return OPTIONS.filter((d) => !d.when || d.when(o)).map((d) => `${d.key}=${o[d.key]}`).join(' ');
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
  if (o.map === 'light') extra.push('Web.LBA.MaxOptKFs: 20', 'Web.LBA.MaxFixedKFs: 20', 'Web.Mapping.Neighbors: 15');
  return extra.length ? `${y.replace(/\s*$/, '\n')}\n# web options (web/options.js)\n${extra.join('\n')}\n` : y;
}

// Renders <label><select> controls into `container`; calls onChange(options) after every change.
export function buildOptionsUI(container, o, onChange = () => {}) {
  container.textContent = '';
  const rows = [];
  for (const d of OPTIONS) {
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
