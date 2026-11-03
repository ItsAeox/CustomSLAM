#!/usr/bin/env bash
# AlvaAR vs WebVIO beyond the benchmarks (user, 3 Nov: run AlvaAR on our own data where it needs no change), same frozen
# build and arms as main_1103: the synthetic set's 42 core sequences (ar / gesture / orbit / rotate / shake / slide /
# walk x 6; exact ground truth) and the 15 Pixel 9 recordings of 18 Oct (no ground truth), seeds / runs 1-2, --rt-wall,
# 4 at a time, run order shuffled, load logged. Waits for main_1103 to finish (one batch at a time on the quiet PC).
set -uo pipefail
R=/mnt/h/Research; REPO=$R/CustomSLAM; O=$R/results/ext_1103; D=$R/datasets
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python; LOAD=$R/logs/ext_1103_load.log
until grep -q "=== done" $R/logs/main_1103.log 2>/dev/null; do sleep 60; done
cd $REPO
PRESETS="fast next beta betagyro mapfirst"
SEQS="$(ls -d $D/synth/{ar,gesture,orbit,rotate,shake,slide,walk}[0-5] | sed 's/^/synth /')"$'
'"$(awk '!/^#/ && NF {print "phone '$D'/phone/"$1}' $D/phone/labels.txt)"
echo "$SEQS" | while read -r kind q; do
  n=$(basename $q); mkdir -p $O/settings/$n
  $NODE scripts/page_presets.mjs --base $q/settings.yaml --name $n $O/settings/$n $PRESETS > /dev/null
  for p in $PRESETS; do f=$O/settings/$n/${n}_page-$p.yaml
    grep -q "^Web.Gyro.Prior: 1" $f && printf '\n# ext_1103.sh: the Pixel 9 camera-IMU offset (as the phone page with imudt=-88)\nWeb.Gyro.TimeOffset: -0.088\n' >> $f; done
done
: > $O/runlist.txt
for s in 1 2; do echo "$SEQS" | while read -r kind q; do
  for a in alvaar wasm-stock page-fast page-next page-beta page-betagyro page-mapfirst; do echo "$a $kind $q $s"; done; done; done >> $O/runlist.txt
$P -c "import random; l=[x for x in open('$O/runlist.txt').read().split('\n') if x]; random.Random(11032).shuffle(l); open('$O/runlist_shuffled.txt','w').write('\n'.join(l)+'\n')"
echo "=== start $(date +%T): $(wc -l < $O/runlist_shuffled.txt) runs"
( while true; do { echo "--- $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) running=$(pgrep -fc 'orbslam3_run|alvaar_run')";
    ps -eo pid,pcpu,rss,etimes,args --sort=-pcpu | head -6 | cut -c1-200; } >> $LOAD; sleep 60; done ) &
LOGGER=$!
xargs -P 4 -L 1 bash scripts/experiments/2026-11-03/ext_job.sh < $O/runlist_shuffled.txt
kill $LOGGER
echo "=== done $(date +%T)"
