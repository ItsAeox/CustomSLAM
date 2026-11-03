#!/usr/bin/env bash
# Main-table batch (claude/PAPER.md §4 VII-A, gap 1-2), quiet PC: one frozen build (snap-84c1f17 = HEAD 84c1f17, tag
# paper-batch-1103), EuRoC 11 + TUM-VI rooms 1-6, seeds 1-5, --rt-wall, 4 runs at a time, run order shuffled across
# arms (fixed seed), machine load logged every 60 s. Arms: native-stock, wasm-stock, page-{fast,next,beta,betagyro,
# mapfirst}; alvaar on EuRoC only (no fisheye model; released bundle unmodified). Ranking rule: claude/EXPERIMENTS.md
# main_1103 (written before the results).
set -uo pipefail
R=/mnt/h/Research; REPO=$R/CustomSLAM; O=$R/results/main_1103; D=$R/datasets; SNAP=/root/build/snap-84c1f17
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python
LOAD=$R/logs/main_1103_load.log
cd $REPO
mkdir -p $O/settings
{ echo "commit $(git rev-parse HEAD)"; sha256sum $SNAP/native/orbslam3_run $SNAP/wasm/orbslam3_run.wasm $SNAP/wasm/orbslam3_run.js \
    $REPO/baselines/alvaar/alvaar_run.mjs $R/ext/AlvaAR/dist/alva_ar.js; date; } > $O/BUILD.txt
$NODE scripts/page_presets.mjs $O/settings fast next beta betagyro mapfirst > $O/settings/page_presets.log
ARMS="native-stock wasm-stock page-fast page-next page-beta page-betagyro page-mapfirst"
: > $O/runlist.txt
for s in 1 2 3 4 5; do
  for q in $D/euroc/*; do for a in $ARMS alvaar; do echo "$a euroc $q $s" >> $O/runlist.txt; done; done
  for k in 1 2 3 4 5 6; do for a in $ARMS; do echo "$a tumvi $D/tumvi/dataset-room${k}_512_16 $s" >> $O/runlist.txt; done; done
done
$P -c "import random; l=open('$O/runlist.txt').read().split('\n'); l=[x for x in l if x]; random.Random(1103).shuffle(l); open('$O/runlist_shuffled.txt','w').write('\n'.join(l)+'\n')"
echo "=== start $(date +%T): $(wc -l < $O/runlist_shuffled.txt) runs"
( while true; do { echo "--- $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) running=$(pgrep -fc 'orbslam3_run|alvaar_run' )";
    ps -eo pid,pcpu,rss,etimes,args --sort=-pcpu | head -6 | cut -c1-200; } >> $LOAD; sleep 60; done ) &
LOGGER=$!
xargs -P 4 -L 1 bash scripts/experiments/2026-11-03/main_job.sh < $O/runlist_shuffled.txt
kill $LOGGER
echo "=== runs done $(date +%T)"
$P eval/collect_results.py > /dev/null && echo "collected"
echo "=== done $(date +%T)"
