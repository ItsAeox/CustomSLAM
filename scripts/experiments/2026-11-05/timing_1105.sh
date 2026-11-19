#!/usr/bin/env bash
# 5 Nov (paper VII-B, PC part; the user: "if you need any PC test, just do it"): runtime of the frozen build, one run at
# a time on a quiet PC. Every timing in the paper's drafts so far came from older builds or from 4-at-a-time batches
# (VI: timing only from one-at-a-time sessions). Frozen snap-84c1f17, --rt-wall as main_1103, MH01 / V1_02 / room2,
# 2 runs each. Arms: native-stock, wasm-stock, page-fast / next / beta (WASM), native-fast / next / beta (same settings,
# native), alvaar on EuRoC. Then score_timing_1105.py -> results/timing_1105/TIMING.md. Load logged every 60 s.
set -uo pipefail
R=/mnt/h/Research; REPO=$R/CustomSLAM; O=$R/results/timing_1105; D=$R/datasets; X=$REPO/scripts/experiments/2026-11-05
P=/opt/slam/venv/bin/python
cd $REPO
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
mkdir -p $O
sha256sum /root/build/snap-84c1f17/native/orbslam3_run /root/build/snap-84c1f17/wasm/orbslam3_run.wasm > $O/BUILD.txt
(while sleep 60; do echo "$(date '+%F %T') $(cut -d' ' -f1 /proc/loadavg)"; done >> $R/logs/timing_1105_load.log) &
LOADPID=$!
: > $O/runlist.txt
for k in 1 2; do
  for q in $D/euroc/MH_01_easy $D/euroc/V1_02_medium; do
    for a in native-stock wasm-stock page-fast page-next page-beta native-fast native-next native-beta alvaar; do
      echo "$a euroc $q $k" >> $O/runlist.txt; done; done
  for a in native-stock wasm-stock page-fast page-next page-beta native-fast native-next native-beta; do
    echo "$a tumvi $D/tumvi/dataset-room2_512_16 $k" >> $O/runlist.txt; done
done
echo "=== start $(date +%T): $(wc -l < $O/runlist.txt) runs"
xargs -P 1 -L 1 bash $X/timing_job.sh < $O/runlist.txt
kill $LOADPID
$P $X/score_timing_1105.py --root $R > $O/TIMING.md 2>&1
echo "=== done $(date +%T)"
