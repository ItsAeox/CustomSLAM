#!/usr/bin/env bash
# rot_screen (2026-10-18): no-regression screen of the pure-rotation prototype (Web.Rot.FarPoints, fast-rot) against
# "Hybrid, fastest" (preset-fast), screen-A protocol: native (pinned snapshot), phone
# emulation (--live --rate-scale 0.5: native drops 24-37 % of frames, close to WASM's 30-43 %), all arms side by side.
# Discriminating subset (where preset-fast loses tracking; TUM-VI rooms have turns on the spot), seeds 1-3. The target
# case (turning in place without sideways motion) needs phone recordings: this only checks that nothing gets worse.
#   preset-fast | fast-rot
# Usage: rot_screen.sh <snapshot tag> <out name> <arm...>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/snap-$1/native/orbslam3_run
O=/mnt/h/Research/results/$2; shift 2
ARMS="$*"
echo "=== start $(date +%T) binary $NATIVE arms $ARMS"
for a in $ARMS; do
  SPECS=()
  for n in V1_03_difficult V2_03_difficult MH_04_difficult MH_05_difficult; do
    SPECS+=("euroc_$n|euroc|$D/euroc/$n|$V/EuRoC_mono_$a.yaml|$G/euroc_$n.txt|")
  done
  for r in 1 2 3 5; do n=dataset-room${r}_512_16; SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$a.yaml|$G/tumvi_$n.txt|--clahe"); done
  g=""; [[ $a == *gyro* ]] && g="--gyro"
  FLAVOURS=native JOBS=2 EXTRA="--live --rate-scale 0.5 $g" bash scripts/seed_sweep.sh $O/$a "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in $ARMS; do
  echo "--- $a"
  $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | tail -1
  cat $O/$a/*/*/track_times.txt | awk '{n++; c[$3]++; s+=$2} END{printf "  frames %d: ok %.3f, lost %.3f, not init %.3f, track ms mean %.2f\n", n, c[2]/n, (c[3]+c[4])/n, c[1]/n, s/n}'
  grep -h "RECOVERY-STATS" $O/$a/*/*/run.log | awk '{for(i=2;i<=NF;i++){split($i,x,"="); s[x[1]]+=x[2]}} END{printf "  coast tries %d, coast recoveries %d, reloc recoveries %d\n", s["coast_tries"], s["coast_recoveries"], s["reloc_recoveries"]}'
  grep -h "ROT-STATS" $O/$a/*/*/run.log | awk '{for(i=2;i<=NF;i++){split($i,x,"="); s[x[1]]+=x[2]}; n++} END{if(n) printf "  far points created %d, culled %d over %d runs\n", s["far_created"], s["far_culled"], n}'
  echo "  maps per run: $(for f in $O/$a/*/*/run.log; do grep -c 'Creation of new map' $f; done | awk '{s+=$1; n++} END{printf "%.2f", s/n/2}')"
  grep -h "KLT-STATS" $O/$a/*/*/run.log | awk '{for(i=2;i<=NF;i++){split($i,x,"="); s[x[1]]+=x[2]}; n++} END{printf "  KLT frame ms %.2f (flow %.2f, pose %.2f)\n", s["klt_ms_mean"]/n, s["flow_ms"]/n, s["pose_ms"]/n}'
  grep -h "FASTPOSE-STATS" $O/$a/*/*/run.log | awk '{for(i=2;i<=NF;i++){split($i,x,"="); s[x[1]]+=x[2]}; n++} END{if(n) printf "  fast pose ms mean %.3f over %d runs\n", s["fast_ms_mean"]/n, n}'
done
B=$(echo $ARMS | cut -d' ' -f1)
for a in $ARMS; do [ $a = $B ] || $P eval/compare_builds.py --across $O/$B $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -1; done
$P eval/screen_table.py $O preset-fast
echo "=== done $(date +%T)"
