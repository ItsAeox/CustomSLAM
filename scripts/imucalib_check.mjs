// Check web/imucalib.js against eval/estimate_time_offset.py --focal on a dataset run (TUM trajectory + EuRoC-format
// IMU csv + the camera->IMU rotation of the settings file). Usage: node scripts/imucalib_check.mjs <traj> <imu.csv> <yaml>
import { readFileSync } from 'node:fs';
const { estimateImuCalib } = await import(new URL('../web/imucalib.js', import.meta.url));
const [trajF, imuF, yamlF] = process.argv.slice(2);
const poses = readFileSync(trajF, 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
  const v = l.trim().split(/\s+/).map(Number); return [v[0], v[4], v[5], v[6], v[7], 1];
});
const gyro = readFileSync(imuF, 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
  const v = l.split(',').map(Number); return [v[0] * 1e-9, v[1], v[2], v[3]];
});
const m = readFileSync(yamlF, 'utf8').match(/IMU\.T_b_c1:[\s\S]*?data:\s*\[([\s\S]*?)\]/)[1].split(',').map(Number);
const Rbc = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
const Rcb = [Rbc[0], Rbc[3], Rbc[6], Rbc[1], Rbc[4], Rbc[7], Rbc[2], Rbc[5], Rbc[8]];
console.log(JSON.stringify(estimateImuCalib(poses, gyro, { Rcb })));
