#!/usr/bin/env bash
# Queue 7 (after queue 6): build the keyframe-policy and BoW-depth knobs (commits 0f27e47, + Web.BoW.LevelsUp), then
#   kf_policy: stock vs Web.KF.RefRatio 0.8 / 0.7 vs Web.Mapping.CullRedundancy 0.8 / 0.7, native, rt-wall, side by side
#     EuRoC 11 + TUM-VI rooms 6 (accuracy, keyframes, mapping work) + magistrale1 (memory, per-map ATE via H3)
#   then H3 re-runs of outdoors1-4 (every map scored), stock native + WASM, rt-wall
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-16_17; L=/mnt/h/Research/logs
until grep -q "=== queue6 done" $L/queue6.log 2>/dev/null; do sleep 120; done
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== queue7 build $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j2 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED"
grep -c "KF.RefRatio" /root/build/slam-native-teblid/orbslam3_run > /dev/null && grep -c "BoW.LevelsUp" /root/build/slam-native-teblid/orbslam3_run > /dev/null \
  && echo "native has the knobs" || { echo "build missing knobs: stop"; exit 1; }
bash $E/snapshot.sh kf1
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/snap-kf1/native/orbslam3_run WASM_JS=/root/build/snap-kf1/wasm/orbslam3_run.js
O=/mnt/h/Research/results/kf_policy
{
  echo "=== kf_policy start $(date +%T)"
  ARMS="stock kf080 kf070 cull080 cull070"
  for a in $ARMS; do
    SPECS=()
    for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$a.yaml|$G/euroc_$n.txt|"); done
    for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$a.yaml|$G/tumvi_$n.txt|--clahe"); done
    FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$a "1 2 3" "${SPECS[@]}" > /dev/null &
  done
  wait
  # magistrale1 (~2 GB each): two arms at a time, never five (WSL has 7.8 GB; ISSUES H2)
  n=dataset-magistrale1_512_16
  set -- $ARMS
  while [ $# -gt 0 ]; do
    for a in $1 ${2:-}; do
      FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$a "1 2 3" \
        "tumvi_magistrale1|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$a.yaml|$G/tumvi_$n.txt|--clahe" > /dev/null &
    done
    wait; shift; [ $# -gt 0 ] && shift
  done
  for f in $O/*/tumvi_magistrale1/*/; do
    $P eval/evaluate_multimap.py --gt $G/tumvi_dataset-magistrale1_512_16.txt --run $f --out $f/metrics_multimap.json
  done
  for a in $ARMS; do
    echo "--- $a"; $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | tail -3
    echo "  keyframes (final, sum) $(awk '/^keyframes/{s+=$2} END{print s}' $O/$a/*/*/run_info.txt), mem_mb median $(awk '/^mem_mb/{print $2}' $O/$a/*/*/run_info.txt | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
    grep -h "LM-STATS" $O/$a/*/*/run.log | awk '{for(i=2;i<=NF;i++){split($i,x,"="); s[x[1]]+=x[2]}} END{printf "  LM: keyframes %d, local BAs %d, aborted %d\n", s["keyframes"], s["lba_done"], s["lba_aborted"]}'
  done
  for a in kf080 kf070 cull080 cull070; do $P eval/compare_builds.py --across $O/stock $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -2; done
  echo "=== kf_policy done $(date +%T)"
} > $L/kf_policy.log 2>&1
O=/mnt/h/Research/results/outdoors_h3
{
  echo "=== outdoors_h3 start $(date +%T)"
  SPECS=()
  for i in 1 2 3 4; do n=dataset-outdoors${i}_512_16; SPECS+=("tumvi_outdoors$i|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_stock.yaml|$G/tumvi_$n.txt|--clahe"); done
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
  for f in $O/stock/*/*/; do
    $P eval/evaluate_multimap.py --gt $G/tumvi_dataset-$(basename $(dirname $f) | sed 's/tumvi_//')_512_16.txt --run $f --out $f/metrics_multimap.json
  done
  echo "=== outdoors_h3 done $(date +%T)"
} > $L/outdoors_h3.log 2>&1
bash $E/vocab_eval2.sh kf1 > $L/vocab_eval.log 2>&1
echo "=== queue7 done $(date +%T)"
