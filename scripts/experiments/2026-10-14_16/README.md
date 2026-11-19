# Experiment scripts, 2026-10-14 to 2026-10-16 (archived as run)

The exact scripts behind the sweeps and builds of these three days, copied unchanged from the working folder
they ran from. They are records, not maintained tools: paths point at that working folder, at the WSL build
directories (`/root/build/...`) and at `/mnt/h/Research/...`. The experiment registry (`H:\Research\claude\EXPERIMENTS.md`)
names the script behind each result; the results themselves are in `H:\Research\results\<experiment>\` and
`results/all_runs.csv` (`eval/collect_results.py`).

Main groups:
- `run_queue2.sh` … `run_queue5.sh`: overnight queues (TEBLID, speed_rt, inertial_rt/inertial_rt2, outdoors_rt,
  magistrale1_diag); `run_*.sh`: single sweeps (m0 parity, sync/half-speed, gates, live emulation, hard10, …).
- `g13_*.sh`: phone IMU freeze reproduction and fix check (phone-like IMU from `scripts/degrade_imu.py`).
- `inertial_sync.sh`, `mag_native_mem.sh`: G14 (speed) and G12 (memory) diagnostics.
- `build_*.sh`, `rebuild_all.sh`: builds (one directory at a time, low parallelism: WSL has 7.8 GB).
- `gt_*.sh`: ground-truth generation (body frame, Magistrale, outdoors); `extract_*.ps1`: dataset extraction.
- `tree_*.py`: export of the AR tree model from the user's Blender file.

One-off source patches applied during these days are not here; they are in the git history of the files they changed.
