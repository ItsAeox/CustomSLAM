#!/usr/bin/env bash
# 2 Nov (task 4, PLAN §0: frames 25-35 ms on the phone vs the 25 ms budget; the map start at 9-12 fps for ~3 s): where the
# tracking thread's time goes with the map-first setup. Profiling build (REGISTER_TIMES: /root/build/slam-wasm-prof, rebuilt
# from the working tree; same flags as the snapshots apart from the timers): per-frame stage times (stage_times.txt: ORB
# extraction, pose prediction, local-map tracking, keyframe, map start: matching / two-view / first map) and LocalMapping's
# per-keyframe stages (LocalMapTimeStats.txt). Synthetic flatfirst0 / flat0 / ar0 / walk0 / orbit0 / rotate0 with
# combo_1102's settings: mapfirst-like (keep_)g005t5_cov2_globall and (keep_)g005t5; live@1.0, seed 1, one run at a time
# (quiet machine). Then V8 CPU profiles (--cpu-prof, every thread) of flatfirst0 mapfirst: wasm functions by self and
# total time (cpuprof_top.py). Usage: prof_1102.sh   (finished runs are skipped)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
B=/root/build/slam-wasm-prof X=scripts/experiments/2026-10-31
echo "=== build prof $(date +%T)"
cmake $B > /dev/null 2>&1
nice -n 10 cmake --build $B -j4 --target orbslam3_run 2>&1 | grep -E "error|FAILED" | head
ls -la --time-style=+%T $B/orbslam3_run.wasm $B/orbslam3_run.js.symbols
# a failed build keeps the old binary: stop unless it is newer than the timer edits
[ $B/orbslam3_run.wasm -nt slam/orbslam3/src/Tracking.cc ] || { echo "=== STALE BUILD, stopping"; exit 1; }
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=$B/orbslam3_run.js
S=/mnt/h/Research/datasets/synth C=/mnt/h/Research/results/combo_1102 O=/mnt/h/Research/results/prof_1102
run() {  # seq arm [out name (default: the arm)] [extra node flags]
  local o=$O/${3:-$2}/$1 st=$C/$1/settings/$1_arm-$2.yaml
  [ -s $o/run_info.txt ] && return
  mkdir -p $o
  $NODE ${4:-} $JS --dataset euroc --mode mono --seq $S/$1 --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $st --seed 1 \
    --live --rate-scale 1.0 --gyro $(cat $C/$1/settings/$1_arm-$2.args) --out $o > $o/run.log 2>&1
  echo "--- $1 $2 $(date +%T)"; /opt/slam/venv/bin/python scripts/experiments/2026-10-18/p95_stages.py $o
}
for q in flatfirst0 flat0 ar0; do for a in keep_g005t5_cov2_globall keep_g005t5; do run $q $a; done; done
for q in walk0 orbit0 rotate0; do for a in g005t5_cov2_globall g005t5; do run $q $a; done; done
/opt/slam/venv/bin/python $X/prof_table.py $O > $O/PROF.md 2>&1
echo "=== cpu profile $(date +%T)"
run flatfirst0 keep_g005t5_cov2_globall keep_g005t5_cov2_globall_cpuprof "--cpu-prof --cpu-prof-dir=$O/keep_g005t5_cov2_globall_cpuprof/flatfirst0/cpuprof"
/opt/slam/venv/bin/python $X/cpuprof_top.py $O/keep_g005t5_cov2_globall_cpuprof/flatfirst0/cpuprof $B/orbslam3_run.js.symbols \
  > $O/CPUPROF.md 2>&1
echo "=== done $(date +%T)"
