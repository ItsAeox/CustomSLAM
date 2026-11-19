#!/usr/bin/env bash
# One run of upstream ORB-SLAM3 monocular on a TUM-VI room (ISSUES G39, user OK 3 Nov evening): v1.0 4452a3c with two
# compile-only edits (-std=c++17; mnFullBAIdx++ on a bool -> = true), its own example (mono_tum_vi: CLAHE 3.0 / 8x8,
# real-time sleep), settings (Examples/Monocular/TUM-VI.yaml), timestamp file and text vocabulary, as released.
# Output: the camera trajectory of the largest map (SaveTrajectoryEuRoC, ns). Scored by our evaluator against the
# camera-frame GT (metrics.json) and the IMU-body / mocap0 GT (metrics_body.json). Usage: <room> <run>
set -uo pipefail
ROOM="$1"; RUN="$2"
R=/mnt/h/Research; U=$R/third_party/ORB_SLAM3; P=/opt/slam/venv/bin/python; E=$R/CustomSLAM/eval
SEQ=dataset-${ROOM}_512_16; O=$R/results/upstream_tumvi_1103/$ROOM/run$RUN
mkdir -p "$O"; cd "$O"
echo "$(date +%T) start $ROOM run$RUN" >> $R/logs/upstream_tumvi_1103_runs.log
timeout 1200 $U/Examples/Monocular/mono_tum_vi $U/Vocabulary/ORBvoc.txt $U/Examples/Monocular/TUM-VI.yaml \
  $R/datasets/tumvi/$SEQ/mav0/cam0/data $U/Examples/Monocular/TUM_TimeStamps/dataset-${ROOM}_512.txt $SEQ > run.log 2>&1
rc=$?; [ $rc -ne 0 ] && echo "run failed (exit $rc)" >> run.log
if [ -s "f_$SEQ.txt" ]; then
  awk '{printf "%.9f %s %s %s %s %s %s %s\n", $1/1e9, $2, $3, $4, $5, $6, $7, $8}' "f_$SEQ.txt" > traj_tum.txt
  $P $E/evaluate.py --gt $R/results/gt/tumvi_$SEQ.txt --est traj_tum.txt --out metrics.json --label "upstream $ROOM run$RUN" > eval.log 2>&1
  $P $E/evaluate.py --gt $R/results/gt/tumvi_${SEQ}_body.txt --est traj_tum.txt --out metrics_body.json --label "upstream body $ROOM run$RUN" >> eval.log 2>&1
fi
echo "$(date +%T) end   $ROOM run$RUN rc=$rc $(tail -1 eval.log 2>/dev/null | cut -c1-120)" >> $R/logs/upstream_tumvi_1103_runs.log
