#!/usr/bin/env bash
# Experiment queue v2 (2026-10-15 evening): honest real-time pacing (--rt-wall: every frame on a wall-clock schedule,
# next image loaded in the background; ISSUES H1), new runners (build_rt.sh), at most ~6 concurrent runs and no
# builds (WSL 7.8 GB, H2). Arms of each comparison run side by side (identical load). Each step logs to its own file;
# run_info.txt of every run records effective_rate (1.0 = real time).
#   1 teblid_eval  ORB (official vocab) vs ORB (own vocab) vs TEBLID (own vocab), native
#   2 live         phone emulation: --live --rate-scale 0.5 and 0.33, stock vs time-scaled motion, WASM
#   3 hard10       stock vs hybrid, 10 seeds, the five hardest sequences, native + WASM
#   4 magistrale   stock (native + WASM) vs hybrid (WASM), 3 seeds, 2 jobs per arm (memory)
#   5 klt_gates    hybrid gate variants vs stock, native
set -uo pipefail
RATES="${RATES:-0.5 0.33}"   # live: frames arrive 2x / 3x faster = a 2x / 3x slower device
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
VOC=/mnt/h/Research/vocab; L=/mnt/h/Research/logs; R=/mnt/h/Research/results; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/slam-native-teblid/orbslam3_run WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
CMP="--bad-ate-mm 200 --bad-rot 1000"
euroc() { for e in $2; do SPECS+=("euroc_$e|euroc|$D/euroc/$e|$V/EuRoC_mono_$1.yaml|$G/euroc_$e.txt|"); done; }
rooms() { for r in $2; do SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/dataset-${r}_512_16|$V/TUM-VI_mono_$1.yaml|$G/tumvi_dataset-${r}_512_16.txt|--clahe"); done; }
rates() { echo "effective playback rate (median over runs) per arm:"; for d in "$@"; do
  echo "  $(basename $d): $(cat $d/*/*/run_info.txt 2>/dev/null | awk '/^effective_rate/{print $2}' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)], "(n=" NR ")"}')"; done; }
ALL_EUROC="$(ls $D/euroc)"; ALL_ROOMS="room1 room2 room3 room4 room5 room6"

# ---- 1. TEBLID evaluation
{
  echo "=== start $(date +%T)"
  O=$R/teblid_eval_rt; EU="MH_01_easy MH_03_medium MH_05_difficult V1_01_easy V1_02_medium V2_02_medium"
  SPECS=(); euroc stock "$EU"; rooms stock "$ALL_ROOMS"
  VOCAB=$VOC/ORBvoc.bin FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/orb-official "1 2 3" "${SPECS[@]}" > /dev/null &
  VOCAB=$VOC/ORBvoc_own_k10L6.bin FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/orb-own "1 2 3" "${SPECS[@]}" > /dev/null &
  SPECS=(); euroc teblid "$EU"; rooms teblid "$ALL_ROOMS"
  VOCAB=$VOC/TEBLIDvoc_own_k10L6.bin FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/teblid-own "1 2 3" "${SPECS[@]}" > /dev/null &
  wait
  for v in orb-official orb-own teblid-own; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v $CMP; done
  $P eval/compare_builds.py --across $O/orb-own $O/teblid-own --a native $CMP
  $P eval/compare_builds.py --across $O/orb-official $O/orb-own --a native $CMP
  rates $O/orb-official $O/orb-own $O/teblid-own
  echo "=== done $(date +%T)"
} > $L/teblid_eval_rt.log 2>&1

# ---- 2. Phone emulation (live)
for RATE in $RATES; do
{
  echo "=== start $(date +%T) rate $RATE"
  O=$R/live_r$RATE
  SPECS=(); euroc stock "$ALL_EUROC"; rooms stock "$ALL_ROOMS"
  FLAVOURS=wasm JOBS=2 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null &
  SPECS=(); euroc tsm "$ALL_EUROC"; rooms tsm "$ALL_ROOMS"
  FLAVOURS=wasm JOBS=2 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/tsm "1 2 3" "${SPECS[@]}" > /dev/null &
  wait
  for v in stock tsm; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v $CMP; done
  $P eval/compare_builds.py --across $O/stock $O/tsm --a wasm $CMP
  for v in stock tsm; do
    echo "  $v dropped fraction: $(cat $O/$v/*/*/run_info.txt | awk '/^frames/{f+=$2} /^dropped/{d+=$2} END{printf "%.3f", d/f}')"
  done
  echo "=== done $(date +%T)"
} > $L/live_r$RATE.log 2>&1
done

# ---- 3. hard10: stock vs hybrid robustness, 10 seeds
{
  echo "=== start $(date +%T)"
  O=$R/hard10_rt; HE="MH_05_difficult V1_03_difficult V2_03_difficult"; HR="room1 room5"
  SPECS=(); euroc stock "$HE"; rooms stock "$HR"
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3 4 5 6 7 8 9 10" "${SPECS[@]}" > /dev/null &
  SPECS=(); euroc klt "$HE"; rooms klt "$HR"
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/klt "1 2 3 4 5 6 7 8 9 10" "${SPECS[@]}" > /dev/null &
  wait
  for fl in wasm native; do $P eval/compare_builds.py --across $O/stock $O/klt --a $fl $CMP; done
  rates $O/stock $O/klt
  echo "=== done $(date +%T)"
} > $L/hard10_rt.log 2>&1

# ---- 4. Magistrale (long: 2 jobs per arm, memory)
{
  echo "=== start $(date +%T)"
  O=$R/magistrale_rt; M=(); for i in 1 2 3 4 5 6; do M+=("dataset-magistrale$i"); done
  mspecs() { SPECS=(); for n in "${M[@]}"; do SPECS+=("tumvi_${n}|tumvi|$D/tumvi/${n}_512_16|$V/TUM-VI_mono_$1.yaml|$G/tumvi_${n}_512_16.txt|--clahe"); done; }
  mspecs stock; FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null &
  mspecs klt;   FLAVOURS=wasm JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/klt "1 2 3" "${SPECS[@]}" > /dev/null &
  wait
  $P eval/summarize_sweep.py $O/stock --bad-ate-mm 1000 --bad-rot 1000
  $P eval/compare_builds.py --across $O/stock $O/klt --a wasm --bad-ate-mm 1000 --bad-rot 1000
  $P eval/compare_builds.py $O/stock --a native --b wasm --bad-ate-mm 1000 --bad-rot 1000
  rates $O/stock $O/klt
  echo "=== done $(date +%T)"
} > $L/magistrale_rt.log 2>&1

# ---- 5. KLT gates
{
  echo "=== start $(date +%T)"
  O=$R/klt_gates_rt; GE="MH_01_easy MH_05_difficult V1_03_difficult V2_01_easy V2_03_difficult"; GR="room1 room2 room5"
  for var in stock klt klt-flow15 klt-frac07 klt-gated; do
    SPECS=(); euroc $var "$GE"; rooms $var "$GR"
    FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$var "1 2 3" "${SPECS[@]}" > /dev/null &
  done
  wait
  for var in klt klt-flow15 klt-frac07 klt-gated; do $P eval/compare_builds.py --across $O/stock $O/$var --a native $CMP; done
  rates $O/stock $O/klt
  echo "=== done $(date +%T)"
} > $L/klt_gates_rt.log 2>&1
echo "queue2 finished $(date +%T)" >> $L/queue2.log
