# WASM (regular build) stock vs KLT on MH01, 1200 frames, fast mode, one after another (same load)
V=/mnt/h/Research/results/settings_variants; O=/mnt/h/Research/results/klt_dev/wasm; mkdir -p $O
N=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
for v in ${VARS:-stock klt}; do
  $N /root/build/slam-wasm/orbslam3_run.js --dataset euroc --mode mono --seq /mnt/h/Research/datasets/euroc/MH_01_easy \
     --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $V/EuRoC_mono_$v.yaml --seed 1 --max-frames 1200 --out $O/$v > $O/$v.log 2>&1
  tt=$(awk '{print $2}' $O/$v/track_times.txt | sort -n | awk '{a[NR]=$1} END {printf "median %.1f p95 %.1f mean %.1f", a[int(NR/2)], a[int(NR*0.95)], 0}')
  mean=$(awk '{s+=$2; n++} END {printf "%.1f", s/n}' $O/$v/track_times.txt)
  echo "$v: track ms $tt (mean $mean) | $(grep -h KLT-STATS $O/$v.log)"
done
