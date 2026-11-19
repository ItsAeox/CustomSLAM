#!/usr/bin/env bash
# 2 Nov (user: "go ahead", PLAN §0): the saved room on the synthetic flat - map once, come back in a later session.
# Session A maps the bedroom first, places the object and saves (flatfirst0 to the end of its 'look' phase, 38.5 s:
# --max-frames 1148, tap at 32.79 s, keep map from the placement, --save-map). Session B, later, loads that map with keep
# map on from the first frame (the page with a restored tree: ar.js keep = anchor 'keep' && cactus) on each flat route
# (flat0-5, same flat, same spot, same camera): 'start' from the first frame (in the bedroom near A's start pose: best
# case) and 'out' from 3 s after leaving the bedroom (--start-at; back in by the route's way). B uses A's settings.
# Arms: arms_savedroom.txt. live@1.0 and phone pace (rate-scale 0.25); A seeds 1-2 (two maps per arm), B seed 1.
# Scoring: eval/anchor_return.py --place floor --refine --save-anchor (A), --anchor-file (B); saved_table.py -> SAVED.md.
# Usage: saved_room_run.sh <snapshot tag (orbslam3_run with --start-at)>   (finished runs are skipped: re-run to resume)
TAG=$1
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31 PY=/opt/slam/venv/bin/python
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node JS=/root/build/snap-$TAG/wasm/orbslam3_run.js V=/mnt/h/Research/vocab
SA=flatfirst0 NA=1148 TAP=32.79
# B start for 'out': 3 s after the route leaves the bedroom (meta.json phases 'out', motion clock), on the frame clock
startat() { $PY -c "
import json, numpy as np
s = '$D/$1'; m = json.load(open(s + '/meta.json'))['synthetic']['motion']['phases_s']
t0 = float(open(s + '/gt/camera_dense.csv').readlines()[1].split()[0])
f0 = np.loadtxt(s + '/mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0, dtype=np.int64)[0] * 1e-9
print(round(m['out'][0] + 3 - (f0 - t0), 3))"; }
run_a() {  # arm seed
  local o=$O/A/$SA/$1/seed$2
  [ -s $o/run_info.txt ] && return; mkdir -p $o
  timeout 1800 $NODE $JS --dataset euroc --mode mono --seq $D/$SA --vocab $V/ORBvoc.bin --settings $O/settings/A_$1.yaml \
    --seed $2 ${MODE:---live --rate-scale 1.0} --gyro --trace --max-frames $NA --keep-map-at $TAP --tap-at $TAP --dump-local \
    --save-map $o/room.map --out $o > $o/run.log 2>&1 || echo "  A $1 seed $2: exit $?"
}
run_b() {  # route mode arm seedA
  local a=$O/A/$SA/$3/seed$4 o=$O/B/$1/$3__$2__A$4/seed1 st=""
  [ -s $o/run_info.txt ] && return
  [ -s $a/room.map ] || { echo "  B $*: no saved map"; return; }
  mkdir -p $o
  [ $2 = out ] && st="--start-at $(startat $1)"
  timeout 1800 $NODE $JS --dataset euroc --mode mono --seq $D/$1 --vocab $V/ORBvoc.bin --settings $O/settings/B_$3.yaml \
    --seed 1 ${MODE:---live --rate-scale 1.0} --gyro --trace --load-map $a/room.map $st --out $o > $o/run.log 2>&1 || echo "  B $* exit $?"
}
export -f run_a run_b startat; export D SA NA TAP PY NODE JS V
for pace in live phone; do
  O=/mnt/h/Research/results/savedroom_1102
  if [ $pace = phone ]; then O=/mnt/h/Research/results/savedroom_1102_phone; export MODE="--live --rate-scale 0.25"; else unset MODE; fi
  export O; mkdir -p $O/settings
  echo "=== $pace start $(date +%T) binary $JS" >> $O/run.log
  ARMS=()
  while IFS='|' read -r name preset extra; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    ARMS+=("$name")
    T=$(mktemp -d)
    $NODE scripts/page_presets.mjs --base $D/$SA/settings.yaml --name s $T $preset > /dev/null
    { cat $T/s_page-$preset.yaml; echo; echo "# saved_room_run.sh: $name"; echo "$extra" | tr ';' '\n' | sed 's/^ *//'; } > $O/settings/A_$name.yaml
    { cat $O/settings/A_$name.yaml; echo "# session B: a restored tree keeps the map from the first frame"; echo "Web.Track.KeepMap: 1"; } > $O/settings/B_$name.yaml
    rm -rf $T
  done < $X/arms_savedroom.txt
  for a in "${ARMS[@]}"; do for s in 1 2; do echo "$a $s"; done; done | xargs -P 4 -L 1 bash -c 'run_a $0 $1' >> $O/run.log 2>&1
  for a in "${ARMS[@]}"; do for s in 1 2; do
    r=$O/A/$SA/$a/seed$s
    [ -s $r/anchor_refine.txt ] || $PY eval/anchor_return.py $r $D/$SA --gt /mnt/h/Research/results/gt/synth_$SA.txt --place floor --refine \
      --save-anchor $r/anchor_saved.json > $r/anchor_refine.txt 2>&1
  done; done
  for r in flat0 flat1 flat2 flat3 flat4 flat5; do for m in start out; do for a in "${ARMS[@]}"; do for s in 1 2; do echo "$r $m $a $s"; done; done; done; done |
    xargs -P 4 -L 1 bash -c 'run_b $0 $1 $2 $3' >> $O/run.log 2>&1
  for o in $O/B/*/*/seed1; do
    r=$(basename $(dirname $(dirname $o))); an=$(basename $(dirname $o)); arm=${an%%__*}; sa=${an##*__A}
    [ -s $o/anchor_saved.txt ] || [ ! -s $O/A/$SA/$arm/seed$sa/anchor_saved.json ] || $PY eval/anchor_return.py $o $D/$r \
      --gt /mnt/h/Research/results/gt/synth_$r.txt --anchor-file $O/A/$SA/$arm/seed$sa/anchor_saved.json > $o/anchor_saved.txt 2>&1
  done
  $PY $X/saved_table.py $O > $O/SAVED.md 2>&1
  echo "=== $pace done $(date +%T)" >> $O/run.log
done
