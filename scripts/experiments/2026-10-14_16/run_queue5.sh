#!/usr/bin/env bash
# Queue v5 (after the -DNDEBUG rebuild, ISSUES C8): honest real time (--rt-wall), <= 4 concurrent runs.
#   1 ndebug_check   assert-enabled snapshot vs NDEBUG build, native + WASM, MH01 + room1, 3 seeds, side by side
#   2 inertial_rt    ORB-SLAM3 mono-inertial native vs WASM (NDEBUG), EuRoC 11 + rooms 6, body-frame GT
#   3 outdoors_rt    TUM-VI outdoors1-4 (user-provided), stock native + WASM
#   4 magistrale1_diag  WASM aborts in loop closing (G12): stock + hybrid, 3 seeds, 2 at a time, memory curve
set -uo pipefail
until grep -q "=== all done" /mnt/h/Research/logs/build_ndebug.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
L=/mnt/h/Research/logs; R=/mnt/h/Research/results; P=/opt/slam/venv/bin/python
NEW_N=/root/build/slam-native-teblid/orbslam3_run; NEW_W=/root/build/slam-wasm-ui/orbslam3_run.js
OLD_N=/root/build/snap-assert/native/orbslam3_run; OLD_W=/root/build/snap-assert/wasm/orbslam3_run.js
CMP="--bad-ate-mm 200 --bad-rot 1000"
lmstats() { for d in "$@"; do
  grep -h "LM-STATS" $d/*/*/run.log 2>/dev/null | awk -v n="$(basename $d)" '{for(i=2;i<=NF;i++){split($i,a,"="); s[a[1]]+=a[2]}} END{printf "  %-12s keyframes %d, local BAs done %d, aborted %d (%.1f%%)\n", n, s["keyframes"], s["lba_done"], s["lba_aborted"], 100*s["lba_aborted"]/(s["lba_done"]+s["lba_aborted"]+1e-9)}'
done; }
rates() { for d in "$@"; do echo "  $(basename $d): effective rate median $(cat $d/*/*/run_info.txt 2>/dev/null | awk '/^effective_rate/{print $2}' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"; done; }
trackms() { for d in "$@"; do echo "  $(basename $d): track ms mean over all frames $(cat $d/*/*/track_times.txt 2>/dev/null | awk '{s+=$2;n++} END{printf "%.2f (n=%d)", s/n, n}')"; done; }

# ---- 1. assert vs NDEBUG
{
  echo "=== start $(date +%T)"
  O=$R/ndebug_check
  SP=("euroc_MH_01_easy|euroc|$D/euroc/MH_01_easy|$V/EuRoC_mono_stock.yaml|$G/euroc_MH_01_easy.txt|"
      "tumvi_dataset-room1|tumvi|$D/tumvi/dataset-room1_512_16|$V/TUM-VI_mono_stock.yaml|$G/tumvi_dataset-room1_512_16.txt|--clahe")
  NATIVE=$OLD_N FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/assert_native "1 2 3" "${SP[@]}" > /dev/null &
  NATIVE=$NEW_N FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/ndebug_native "1 2 3" "${SP[@]}" > /dev/null &
  WASM_JS=$OLD_W FLAVOURS=wasm JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/assert_wasm "1 2 3" "${SP[@]}" > /dev/null &
  WASM_JS=$NEW_W FLAVOURS=wasm JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/ndebug_wasm "1 2 3" "${SP[@]}" > /dev/null &
  wait
  for v in assert_native ndebug_native assert_wasm ndebug_wasm; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v $CMP; done
  echo "tracking time:"; trackms $O/assert_native $O/ndebug_native $O/assert_wasm $O/ndebug_wasm
  echo "mapping thread:"; lmstats $O/assert_native $O/ndebug_native $O/assert_wasm $O/ndebug_wasm
  rates $O/assert_native $O/ndebug_native $O/assert_wasm $O/ndebug_wasm
  echo "=== done $(date +%T)"
} > $L/ndebug_check.log 2>&1

export NATIVE=$NEW_N WASM_JS=$NEW_W
# ---- 2. mono-inertial native vs WASM
{
  echo "=== start $(date +%T)"
  O=$R/inertial_rt2; SET=/mnt/h/Research/CustomSLAM/baselines/orbslam3/settings; SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$SET/EuRoC_mono_inertial.yaml|$G/euroc_${n}_body.txt|--mode mono_inertial"); done
  for r in room1 room2 room3 room4 room5 room6; do n=dataset-${r}_512_16
    SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/$n|$SET/TUM-VI_mono_inertial.yaml|$G/tumvi_${n}_body.txt|--clahe --mode mono_inertial"); done
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
  $P eval/summarize_sweep.py $O/stock $CMP
  $P eval/compare_builds.py $O/stock --a native --b wasm $CMP
  grep -l "Assertion\|Aborted" $O/stock/*/*/run.log 2>/dev/null | sed 's/^/  aborted: /'
  rates $O/stock
  echo "=== done $(date +%T)"
} > $L/inertial_rt2.log 2>&1

# ---- 3. outdoors
{
  echo "=== start $(date +%T)"
  bash /mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/gt_outdoors.sh
  O=$R/outdoors_rt; SPECS=()
  for i in 1 2 3 4; do n=dataset-outdoors${i}_512_16
    [ -s $G/tumvi_$n.txt ] && SPECS+=("tumvi_outdoors$i|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_stock.yaml|$G/tumvi_$n.txt|--clahe"); done
  echo "sequences with GT: ${#SPECS[@]}"
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
  $P eval/summarize_sweep.py $O/stock --bad-ate-mm 1000 --bad-rot 1000
  $P eval/compare_builds.py $O/stock --a native --b wasm --bad-ate-mm 1000 --bad-rot 1000
  rates $O/stock
  echo "=== done $(date +%T)"
} > $L/outdoors_rt.log 2>&1

# ---- 4. magistrale1 abort diagnostic
{
  echo "=== start $(date +%T)"
  O=$R/magistrale1_diag; n=dataset-magistrale1_512_16
  for v in stock klt; do
    FLAVOURS=wasm JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$v "1 2 3" \
      "tumvi_magistrale1|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$v.yaml|$G/tumvi_$n.txt|--clahe" > /dev/null &
  done
  wait
  for f in $O/*/*/*/run.log; do
    echo "--- $f"
    grep -E "^frame|Loop detected|Merge|Abort|run failed" "$f" | grep -v GROWABLE | tail -6 | cut -c1-170
    grep -E "^(map_points|keyframes|maps|mem_mb|effective_rate) " "$(dirname $f)/run_info.txt" 2>/dev/null | tr '\n' ' '; echo
  done
  echo "=== done $(date +%T)"
} > $L/magistrale1_diag.log 2>&1
echo "queue5 finished $(date +%T)" >> $L/queue5.log
