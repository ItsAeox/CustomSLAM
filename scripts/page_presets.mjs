// Settings variants exactly as the AR page builds them (web/options.js applyPreset + applyOptions with the AR page
// defaults), so dataset runs test the presets the phone runs. Writes <out>/<EuRoC|TUM-VI>_mono_page-<key>.yaml.
// Gyro presets also need the dataset's IMU.T_b_c1 (the page appends the phone's): copied from *_inertial.yaml.
// Usage: node scripts/page_presets.mjs <out dir> [preset ...]
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const repo = new URL('..', import.meta.url);
const { PRESETS, applyPreset, applyOptions } = await import(new URL('web/options.js', repo));
const AR_DEFAULTS = { reloc: '10', tracker: 'klt' };  // web/ar.js
const [out, ...only] = process.argv.slice(2);
for (const name of ['EuRoC_mono', 'TUM-VI_mono']) {
  const base = readFileSync(new URL(`baselines/orbslam3/settings/${name}.yaml`, repo), 'utf8');
  const inertial = readFileSync(new URL(`baselines/orbslam3/settings/${name}_inertial.yaml`, repo), 'utf8');
  const tbc = inertial.match(/IMU\.T_b_c1:\s*!!opencv-matrix[\s\S]*?\]/)[0];
  for (const p of PRESETS) {
    if (only.length && !only.includes(p.key)) continue;
    const o = applyPreset({}, p, AR_DEFAULTS, 'ar');
    if (o.imu === 'on') continue;
    let y = applyOptions(base, o);
    if (/Web\.Gyro\.Prior: 1/.test(y)) y += `\n# camera -> IMU (dataset; the AR page appends the phone's)\n${tbc}\n`;
    writeFileSync(`${out}/${name}_page-${p.key}.yaml`, y);
    console.log(`${name}_page-${p.key}.yaml: ${(y.match(/^Web\.[^\n]*/gm) || []).join(', ')}`);
  }
}
