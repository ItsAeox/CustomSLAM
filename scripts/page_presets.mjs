// Settings variants exactly as the AR page builds them (web/options.js applyPreset + applyOptions with the AR page
// defaults), so dataset runs test the presets the phone runs. Writes <out>/<EuRoC|TUM-VI>_mono_page-<key>.yaml.
// Gyro presets also need the dataset's IMU.T_b_c1 (the page appends the phone's): copied from *_inertial.yaml.
// Usage: node scripts/page_presets.mjs <out dir> [preset ...]
//        node scripts/page_presets.mjs --base <recording settings.yaml> --name <prefix> <out dir> [preset ...]
//          (phone recordings, record.html: their own camera + IMU block; writes <out>/<prefix>_page-<key>.yaml)
import { readFileSync, writeFileSync } from 'node:fs';
const repo = new URL('..', import.meta.url);
const { PRESETS, applyPreset, applyOptions } = await import(new URL('web/options.js', repo));
const AR_DEFAULTS = { reloc: '10', tracker: 'klt' };  // web/ar.js
const argv = process.argv.slice(2);
const flag = (f) => { const k = argv.indexOf(f); if (k < 0) return null; const v = argv[k + 1]; argv.splice(k, 2); return v; };
const recBase = flag('--base'), recName = flag('--name');
const [out, ...only] = argv;
for (const name of recBase ? [recName || 'rec'] : ['EuRoC_mono', 'TUM-VI_mono']) {
  const base = readFileSync(recBase || new URL(`baselines/orbslam3/settings/${name}.yaml`, repo), 'utf8');
  const tbc = recBase ? null  // a recording's settings already carry the phone IMU block
    : readFileSync(new URL(`baselines/orbslam3/settings/${name}_inertial.yaml`, repo), 'utf8').match(/IMU\.T_b_c1:\s*!!opencv-matrix[\s\S]*?\]/)[0];
  for (const p of PRESETS) {
    if (only.length && !only.includes(p.key)) continue;
    const o = applyPreset({}, p, AR_DEFAULTS, 'ar');
    if (o.imu === 'on') continue;
    let y = applyOptions(base, o);
    if (tbc && /Web\.Gyro\.Prior: 1/.test(y)) y += `\n# camera -> IMU (dataset; the AR page appends the phone's)\n${tbc}\n`;
    writeFileSync(`${out}/${name}_page-${p.key}.yaml`, y);
    console.log(`${name}_page-${p.key}.yaml: ${(y.match(/^Web\.[^\n]*/gm) || []).join(', ')}`);
  }
}
