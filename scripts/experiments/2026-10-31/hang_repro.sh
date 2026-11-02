#!/usr/bin/env bash
# ISSUES G23 / G33: run hang_repro.cc (orbslam3_run's thread layout, 2 threads started + joined per iteration as the map
# start's two-view tries) many times at once and catch a stall: out.log not growing for 30 s while node lives ->
# gdb with hang_gdb.py (gdb.txt; perf.map too when node runs with a perf flag), then kill. Built with orbslam3_run's
# WASM link flags (slam/CMakeLists.txt) into /root/build/hang_repro. Usage: hang_repro.sh <out dir> <parallel copies>
#   <runs> <iterations per run> <mode 0 = new threads | 1 = pool> ["node flags"]
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
X=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31 B=/root/build/hang_repro
O=$1 PAR=$2 RUNS=$3 IT=$4 MODE=$5 NF=${6:-}
mkdir -p $B $O
if ! [ $B/hang_repro.js -nt $X/hang_repro.cc ]; then
  em++ -O3 -pthread -std=c++17 $X/hang_repro.cc -o $B/hang_repro.js -sENVIRONMENT=node,worker -sNODERAWFS=1 \
    -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=8 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=512MB \
    -sSTACK_SIZE=8MB -sDEFAULT_PTHREAD_STACK_SIZE=8MB -sEXIT_RUNTIME=1 --emit-symbol-map || exit 1
fi
export O IT MODE NF X B
one() {
  local o=$O/r$1 pid last=-1 still=0 sz
  [ -s $o/done.txt ] && return
  mkdir -p $o
  /opt/slam/emsdk/node/24.19.0_64bit/bin/node $NF $B/hang_repro.js $IT $MODE $o/trace.txt > $o/out.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 2
    sz=$(stat -c %s $o/out.log)
    if [ "$sz" = "$last" ]; then still=$((still + 2)); else still=0; last=$sz; fi
    if [ $still -ge 30 ]; then
      echo "STALL r$1 at $(date +%T): $(tail -n 2 $o/out.log | tr '\n' ' ')"
      timeout 300 gdb -p $pid -batch -x $X/hang_gdb.py > $o/gdb.txt 2>&1
      [ -f /tmp/perf-$pid.map ] && cp /tmp/perf-$pid.map $o/perf.map
      kill -9 $pid; break
    fi
  done
  wait $pid 2>/dev/null
  rm -f /tmp/perf-$pid.map
  tail -n 1 $o/out.log > $o/done.txt
}
export -f one
echo "=== $(date +%T): mode $MODE, $RUNS runs x $IT iterations, $PAR at once, node flags '$NF'" | tee -a $O/stress.log
seq 1 $RUNS | xargs -P $PAR -I{} bash -c 'one {}' | tee -a $O/stress.log
echo "=== $(date +%T): $(grep -c "^STALL" $O/stress.log) stalls; $(grep -l "^done" $O/r*/done.txt | wc -l) of $RUNS finished" \
  | tee -a $O/stress.log
