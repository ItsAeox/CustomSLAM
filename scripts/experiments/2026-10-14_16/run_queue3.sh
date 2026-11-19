#!/usr/bin/env bash
# Experiment queue v3 (queued after queue v2; honest real time --rt-wall, new runners, <= 6 concurrent runs, no builds):
#   1 speed_rt   stock vs lba5 (local BA 5 LM iterations) vs light-orb (0.6x features, 6 levels), WASM, side by side,
#                EuRoC 11 + rooms 6, 3 seeds. Speed proxy: LM-STATS (local BAs aborted by a new keyframe = mapping
#                thread falling behind), track ms; accuracy: ATE, bad runs.
#   1b inertial  mono-inertial native vs WASM (EuRoC 11 + rooms 6, body-frame GT)
#   2 outdoors   TUM-VI outdoors1-4 (user-provided), stock, native + WASM, 3 seeds (GT only at start/end in the mocap
#                room, like Magistrale: drift over the walk). Needs the extraction + GT (gt_outdoors.sh) first.
set -uo pipefail
until grep -q "queue2 finished" /mnt/h/Research/logs/queue2.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
# ---- 0. builds, nothing else running (WSL 7.8 GB: -j2): runners with map size + memory in run_info (native, WASM),
#         browser module (new bindings: profile, setKeepMap, Calibrator) and the profiling module (web/build-prof)
{
  echo "=== builds $(date +%T)"
  source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
  nice -n 10 cmake --build /root/build/slam-native-teblid -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED"
  nice -n 10 cmake --build /root/build/slam-wasm-ui -j2 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED"
  nice -n 10 cmake --build /root/build/slam-wasm-prof -j2 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED"
  ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run /root/build/slam-wasm-ui/orbslam3_run.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-prof/orbslam3.wasm
  cp /root/build/slam-wasm-ui/orbslam3.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-ui/orbslam3.worker.mjs web/build/
  mkdir -p web/build-prof && cp /root/build/slam-wasm-prof/orbslam3.js /root/build/slam-wasm-prof/orbslam3.wasm /root/build/slam-wasm-prof/orbslam3.worker.mjs web/build-prof/
  echo "=== builds done $(date +%T)"
} > /mnt/h/Research/logs/builds_q3.log 2>&1
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
L=/mnt/h/Research/logs; R=/mnt/h/Research/results; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/slam-native-teblid/orbslam3_run WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
CMP="--bad-ate-mm 200 --bad-rot 1000"
lmstats() { for d in "$@"; do
  grep -h "LM-STATS" $d/*/*/run.log 2>/dev/null | awk -v n="$(basename $d)" '{for(i=2;i<=NF;i++){split($i,a,"="); s[a[1]]+=a[2]}} END{printf "  %-10s keyframes %d, local BAs done %d, aborted %d (%.1f%%), skipped %d\n", n, s["keyframes"], s["lba_done"], s["lba_aborted"], 100*s["lba_aborted"]/(s["lba_done"]+s["lba_aborted"]+1e-9), s["lba_skipped_newkf"]}'
done; }
rates() { for d in "$@"; do echo "  $(basename $d): effective rate median $(cat $d/*/*/run_info.txt 2>/dev/null | awk '/^effective_rate/{print $2}' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"; done; }
specs() {
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$1.yaml|$G/euroc_$n.txt|"); done
  for r in room1 room2 room3 room4 room5 room6; do SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/dataset-${r}_512_16|$V/TUM-VI_mono_$1.yaml|$G/tumvi_dataset-${r}_512_16.txt|--clahe"); done
}

# ---- 1. speed variants
{
  echo "=== start $(date +%T)"
  O=$R/speed_rt
  for v in stock lba5 light-orb; do
    specs $v; FLAVOURS=wasm JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$v "1 2 3" "${SPECS[@]}" > /dev/null &
  done
  wait
  for v in stock lba5 light-orb; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v $CMP; done
  for v in lba5 light-orb; do $P eval/compare_builds.py --across $O/stock $O/$v --a wasm $CMP; done
  echo "mapping thread (LM-STATS, all runs):"; lmstats $O/stock $O/lba5 $O/light-orb
  rates $O/stock $O/lba5 $O/light-orb
  echo "=== done $(date +%T)"
} > $L/speed_rt.log 2>&1

# ---- 1b. mono-inertial (ORB-SLAM3's IMU path in the web port: does WASM match native?), body-frame GT
{
  echo "=== start $(date +%T)"
  O=$R/inertial_rt; SET=/mnt/h/Research/CustomSLAM/baselines/orbslam3/settings; SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$SET/EuRoC_mono_inertial.yaml|$G/euroc_${n}_body.txt|--mode mono_inertial"); done
  for r in room1 room2 room3 room4 room5 room6; do n=dataset-${r}_512_16
    SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/$n|$SET/TUM-VI_mono_inertial.yaml|$G/tumvi_${n}_body.txt|--clahe --mode mono_inertial"); done
  FLAVOURS="native wasm" JOBS=3 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
  $P eval/summarize_sweep.py $O/stock $CMP
  $P eval/compare_builds.py $O/stock --a native --b wasm $CMP
  rates $O/stock
  echo "=== done $(date +%T)"
} > $L/inertial_rt.log 2>&1

# ---- 2. outdoors (user-provided TUM-VI outdoors1-4)
{
  echo "=== start $(date +%T)"
  until grep -q "ALL DONE" $L/extract_outdoors.log 2>/dev/null; do sleep 60; done
  bash /mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/gt_outdoors.sh
  O=$R/outdoors_rt; SPECS=()
  for i in 1 2 3 4; do n=dataset-outdoors${i}_512_16
    [ -s $G/tumvi_$n.txt ] && SPECS+=("tumvi_outdoors$i|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_stock.yaml|$G/tumvi_$n.txt|--clahe"); done
  echo "sequences with GT: ${#SPECS[@]}"
  FLAVOURS="native wasm" JOBS=3 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
  $P eval/summarize_sweep.py $O/stock --bad-ate-mm 1000 --bad-rot 1000
  $P eval/compare_builds.py $O/stock --a native --b wasm --bad-ate-mm 1000 --bad-rot 1000
  rates $O/stock
  echo "=== done $(date +%T)"
} > $L/outdoors_rt.log 2>&1
echo "queue3 finished $(date +%T)" >> $L/queue3.log
