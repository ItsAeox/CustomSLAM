#!/usr/bin/env bash
# One run of timing_1105.sh: <arm> <dataset> <sequence dir> <run>. Same frozen build, settings and pacing (--rt-wall) as
# main_1103 (main_job.sh), but one run at a time, so per-frame times are not shared with other runs. Arms:
# native-stock / wasm-stock (ORB-SLAM3's settings), page-<preset> (WASM, the page's settings) and native-<preset> (the
# same settings, native build: the runtime's cost per preset), alvaar (released bundle; its per-frame time includes its
# mapping, which runs inline). /usr/bin/time -v gives CPU time over all threads and peak memory.
set -uo pipefail
ARM="$1"; DS="$2"; SEQ="$3"; RUN="$4"
R=/mnt/h/Research; REPO=$R/CustomSLAM; O0=$R/results/timing_1105
SNAP=/root/build/snap-84c1f17; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
S=$REPO/baselines/orbslam3/settings; V=$R/results/main_1103/settings
n=$(basename "$SEQ"); NAME=${DS}_$n
if [ "$DS" = euroc ]; then BASE=EuRoC_mono; C=""; else BASE=TUM-VI_mono; C=--clahe; fi
case "$ARM" in
  native-stock) FL=native; ST=$S/$BASE.yaml; CMD=($SNAP/native/orbslam3_run) ;;
  wasm-stock)   FL=wasm;   ST=$S/$BASE.yaml; CMD=($NODE $SNAP/wasm/orbslam3_run.js) ;;
  page-*)       FL=wasm;   ST=$V/${BASE}_$ARM.yaml; CMD=($NODE $SNAP/wasm/orbslam3_run.js) ;;
  native-*)     FL=native; ST=$V/${BASE}_page-${ARM#native-}.yaml; CMD=($SNAP/native/orbslam3_run) ;;
  alvaar)       FL=wasm;   ST=$S/$BASE.yaml; CMD=($NODE $REPO/baselines/alvaar/alvaar_run.mjs) ;;
esac
O=$O0/$ARM/$NAME/run$RUN
mkdir -p "$O"
X=(--rt-wall)
if [ "$ARM" != alvaar ]; then
  X+=(--trace); [ -n "$C" ] && X+=($C)
  grep -qE "^Web\.(Gyro|Scale)\." "$ST" && X+=(--gyro)
fi
echo "$(date +%T) start $ARM $NAME run$RUN $(cut -d' ' -f1 /proc/loadavg)" >> $R/logs/timing_1105_runs.log
timeout -k 30 1200 /usr/bin/time -v -o "$O/time.txt" "${CMD[@]}" --dataset $DS --mode mono --seq "$SEQ" \
  --vocab $R/vocab/ORBvoc.bin --settings "$ST" --out "$O" --seed $RUN "${X[@]}" > "$O/run.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "run failed (exit $rc)" >> "$O/run.log"
echo "$(date +%T) end   $ARM $NAME run$RUN rc=$rc" >> $R/logs/timing_1105_runs.log
