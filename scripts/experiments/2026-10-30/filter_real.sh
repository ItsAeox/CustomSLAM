#!/usr/bin/env bash
# The C++ scale filter (WebScaleFilter.h efa845e, via eval/scale_filter_replay.cpp) on the real phone recordings with a known
# motion: rec_20261029_202938 (brisk 1.00 m slides; true scale per run from scale_known_motion.py: 1.933 / 2.043 / 2.190 m/unit,
# EXPERIMENTS scale_1029) and rec_20261021_070221 (slow 1.00 m slides; scale unobservable for every estimator), the
# beta_scale runs of 29 / 21 Oct (live poses in trace.txt), IMU mav0/imu0 at -88 ms. Prints, per run: first frame with the
# reported 1-sigma < 5 %, the converged estimates (first, median, last), and the window estimator's converged value (trace).
cd /mnt/h/Research/CustomSLAM
g++ -O2 -std=c++17 -I/usr/include/eigen3 -Islam/orbslam3/include eval/scale_filter_replay.cpp -o /root/build/scale_filter_replay 2>&1 | grep error
O=/mnt/h/Research/results/filter_real_1030; mkdir -p $O
for rec in rec_20261029_202938 rec_20261021_070221; do
  R=/mnt/h/Research/results/scale_10${rec:10:2}/$rec/beta_scale; [ $rec = rec_20261021_070221 ] && R=/mnt/h/Research/results/scale_1021/$rec/beta_scale
  d=/mnt/h/Research/datasets/phone/$rec
  for s in 1 2 3; do
    r=$R/seed$s; [ -f $r/trace.txt ] || continue
    /root/build/scale_filter_replay $r $d $d/mav0/imu0/data.csv -88 $O/${rec}_seed$s.txt
    awk -v name="$rec seed$s" 'NR==1 {next} {t[NR]=$1} NR==2 {t0=$1}
         $3>0 && $3<0.05 {n++; v[n]=$2; if (!f) f=$1-t0}
         END {if (!n) {print name ": filter never converged"; exit}
              asort(v, w); printf "%s: filter converged after %.1f s, %d frames; first %.3f, median %.3f, last %.3f m/unit\n", name, f, n, v[1], w[int((n+1)/2)], v[n]}' $O/${rec}_seed$s.txt
    awk -v name="$rec seed$s" 'NF>=29 && $28>0 && $28<0.05 {n++; v[n]=$27} END {if (!n) {print "   window estimator: never < 5 %"; exit} asort(v,w); printf "   window estimator: %d frames < 5 %%, median %.3f m/unit\n", n, w[int((n+1)/2)]}' $r/trace.txt
  done
done
