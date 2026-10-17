// Check web/imuscale.js against eval/scale_from_accel.py on a dataset run (TUM trajectory + EuRoC-format IMU csv + the
// camera->IMU extrinsic of the settings file). Usage: node scripts/imuscale_check.mjs <traj> <imu.csv> <yaml> [window s]
import { readFileSync } from 'node:fs';
const { estimateScale } = await import(new URL('../web/imuscale.js', import.meta.url));
const [trajF, imuF, yamlF, win] = process.argv.slice(2);
const poses = readFileSync(trajF, 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
  const v = l.trim().split(/\s+/).map(Number); return [v[0], v[4], v[5], v[6], v[7], 1, v[1], v[2], v[3]];
}).sort((a, b) => a[0] - b[0]);
const acc = readFileSync(imuF, 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
  const v = l.split(','); return [Number(BigInt(v[0]) / 1000n) * 1e-6, Number(v[4]), Number(v[5]), Number(v[6])];
});
const m = readFileSync(yamlF, 'utf8').match(/IMU\.T_b_c1:[\s\S]*?data:\s*\[([\s\S]*?)\]/)[1].split(',').map(Number);
const Rbc = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
const Rcb = [Rbc[0], Rbc[3], Rbc[6], Rbc[1], Rbc[4], Rbc[7], Rbc[2], Rbc[5], Rbc[8]];
const t0 = performance.now();
const r = estimateScale(poses, acc, { Rcb, window: Number(win || 2) });
console.log(JSON.stringify({ ...r, ms: Math.round(performance.now() - t0) }));
