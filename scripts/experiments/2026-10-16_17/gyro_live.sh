#!/usr/bin/env bash
# gyro_live: does the gyro rotation prior (Web.Gyro.Prior) reduce tracking failures under phone emulation?
# WASM --live --rate-scale 0.5 (late frames dropped), TUM-VI rooms 1-6 + the four hardest EuRoC, seeds 1-3.
# Two sweeps of three arms side by side (identical load within a sweep):
#   A: stock          | stock + gyro (dataset 200 Hz)  | stock + phone-like gyro
#   B: preset-fast    | preset-fast + gyro             | preset-fast + phone-like gyro
# Phone-like gyro: scripts/degrade_imu.py --rate 60 --average --jitter-ms 5 --offset-ms 30 (60 Hz, anti-aliased,
# stamps 30 ms late and jittered), not compensated (Web.Gyro.TimeOffset 0): the prior extrapolates <= 50 ms.
# Usage: gyro_live.sh <snapshot tag>   (binaries with the gyro prior; waits for presets_live to finish)
set -uo pipefail
TAG=$1
until grep -q "=== done" /mnt/h/Research/logs/presets_live.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; S=/mnt/h/Research/datasets_synth; G=/mnt/h/Research/results/gt
V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
export WASM_JS=/root/build/snap-$TAG/wasm/orbslam3_run.js
O=/mnt/h/Research/results/gyro_live; RATE=0.5
EU="MH_04_difficult MH_05_difficult V1_03_difficult V2_03_difficult"; ROOMS="1 2 3 4 5 6"
echo "=== start $(date +%T) binary $WASM_JS"
for n in $EU; do $P scripts/degrade_imu.py $D/euroc/$n $S/euroc/${n}_phonegyro --rate 60 --average --jitter-ms 5 --offset-ms 30; done
for r in $ROOMS; do n=dataset-room${r}_512_16
  $P scripts/degrade_imu.py $D/tumvi/$n $S/tumvi/${n}_phonegyro --rate 60 --average --jitter-ms 5 --offset-ms 30; done
specs() {  # settings variant, data: orig | phone
  SPECS=()
  for n in $EU; do s=$D/euroc/$n; [ $2 = phone ] && s=$S/euroc/${n}_phonegyro
    SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$1.yaml|$G/euroc_$n.txt|"); done
  for r in $ROOMS; do n=dataset-room${r}_512_16; s=$D/tumvi/$n; [ $2 = phone ] && s=$S/tumvi/${n}_phonegyro
    SPECS+=("tumvi_dataset-room$r|tumvi|$s|$V/TUM-VI_mono_$1.yaml|$G/tumvi_$n.txt|--clahe"); done
}
for base in stock preset-fast; do
  specs $base orig;       FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/$base "1 2 3" "${SPECS[@]}" > /dev/null &
  specs $base-gyro orig;  FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale $RATE --gyro" bash scripts/seed_sweep.sh $O/$base-gyro "1 2 3" "${SPECS[@]}" > /dev/null &
  specs $base-gyro phone; FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale $RATE --gyro" bash scripts/seed_sweep.sh $O/$base-gyrophone "1 2 3" "${SPECS[@]}" > /dev/null &
  wait
  echo "  $base arms done $(date +%T)"
done
echo "=== summaries $(date +%T)"
for a in stock stock-gyro stock-gyrophone preset-fast preset-fast-gyro preset-fast-gyrophone; do
  bad=$($P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="wasm"{b+=$5; n+=$3} END{print b "/" n}')
  echo "  $a: bad $bad"
done
for base in stock preset-fast; do
  $P eval/compare_builds.py --across $O/$base $O/$base-gyro --a wasm --bad-ate-mm 200 --bad-rot 1000 | tail -2
  $P eval/compare_builds.py --across $O/$base $O/$base-gyrophone --a wasm --bad-ate-mm 200 --bad-rot 1000 | tail -2
done
echo "=== done $(date +%T)"
