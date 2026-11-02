#!/usr/bin/env bash
# ISSUES G23 / G33: replays that stopped at ~0 % CPU until the 1800 s timeout (1 in ~50-200 runs). Re-run the three
# known cases many times and catch a stall: no growth of run.log + trace.txt for 90 s while the process lives ->
# native: gdb "thread apply all bt" (gdb.txt); WASM: a Node report (SIGUSR2: the JS + wasm stack of every thread,
# report.*.json; name the wasm-function[N] frames with hang_stacks.py and the snapshot's orbslam3_run.js.symbols).
#   lf  = loopfast_1102 flatmap2 keep_g005t5_mi15_loop2 (WASM seed 2 hung at frame ~740)
#   kfq = newarea_1031 longar2 kfq2 (WASM seed 1 hung at frame ~2036)
#   g23 = replays_1018_lbafast rec_20261018_185134 grecover_dt_keep (WASM seed 2 hung at frame ~200)
# Settings and runner arguments are copied from those batches. Usage: hang_stress.sh <snapshot tag> <native|wasm>
#   <repeats> <par> [rate-scale=1.0] [out dir]   (repeat r uses seed r)
TAG=$1 KIND=$2 REPS=$3 PAR=$4 RS=${5:-1.0}
O=${6:-/mnt/h/Research/results/hang_1102_${KIND}_rs$RS}
cd /mnt/h/Research/CustomSLAM
R=/mnt/h/Research/results S=/mnt/h/Research/datasets
mkdir -p $O/settings
cp $R/loopfast_1102/flatmap2/settings/flatmap2_arm-keep_g005t5_mi15_loop2.{yaml,args} $O/settings/ 2>/dev/null
cp $R/newarea_1031/longar2/settings/longar2_arm-kfq2.{yaml,args} $O/settings/ 2>/dev/null
cp $R/replays_1018_lbafast/rec_20261018_185134/settings/rec_20261018_185134_arm-grecover_dt_keep.{yaml,args} $O/settings/ 2>/dev/null
export O KIND RS TAG
run_one() {  # name seq settings-stem rep
  local o=$O/$1/r$4 st=$O/settings/$3.yaml aa="" g="" pid last=-1 still=0 sz
  [ -s $o/done.txt ] && return
  mkdir -p $o
  [ -f $O/settings/$3.args ] && aa=$(cat $O/settings/$3.args)
  grep -qE "Web.Gyro.Prior: 1|Web.Scale.Online: 1" $st && g="--gyro"
  local args="--dataset euroc --mode mono --seq $2 --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $st --seed $4 --live --rate-scale $RS $g --trace $aa --out $o"
  if [ $KIND = native ]; then /root/build/snap-$TAG/native/orbslam3_run $args > $o/run.log 2>&1 &
  else /opt/slam/emsdk/node/24.19.0_64bit/bin/node --report-on-signal --report-directory=$o /root/build/snap-$TAG/wasm/orbslam3_run.js $args > $o/run.log 2>&1 &
  fi
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    sz=$(stat -c %s $o/run.log $o/trace.txt 2>/dev/null | tr '\n' +)
    if [ "$sz" = "$last" ]; then still=$((still + 5)); else still=0; last=$sz; fi
    if [ $still -ge 90 ]; then
      echo "STALL $1 r$4 at $(date +%T): $(tail -c 200 $o/run.log | tr '\n' ' ')"
      if [ $KIND = native ]; then timeout 120 gdb -p $pid -batch -ex "thread apply all bt" > $o/gdb.txt 2>&1
      else kill -USR2 $pid; sleep 20; fi
      kill -9 $pid; echo stall > $o/done.txt; return
    fi
  done
  wait $pid; echo "exit $?" > $o/done.txt
}
export -f run_one
for r in $(seq 1 $REPS); do
  echo "lf $S/synth/flatmap2 flatmap2_arm-keep_g005t5_mi15_loop2 $r"
  echo "kfq $S/synth/longar2 longar2_arm-kfq2 $r"
  echo "g23 $S/phone/rec_20261018_185134 rec_20261018_185134_arm-grecover_dt_keep $r"
done | xargs -P $PAR -L 1 bash -c 'run_one $0 $1 $2 $3' >> $O/stress.log 2>&1
echo "=== $(date +%T): $(grep -l stall $O/*/r*/done.txt 2>/dev/null | wc -l) stalls in $(ls $O/*/r*/done.txt | wc -l) runs" >> $O/stress.log
