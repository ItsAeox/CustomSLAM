#!/usr/bin/env bash
# TEBLID evaluation (queued after the vocabularies and the follow-up sweep): native, real time, seeds 1-3.
#   (TUM-VI: the six rooms; magistrale has its own sweep)
#   orb-official: stock settings + ORBvoc.bin | orb-own: stock + ORBvoc_own | teblid-own: Web.Descriptor 1 + TEBLIDvoc_own
set -uo pipefail
until grep -q "=== vocabularies done" /mnt/h/Research/logs/teblid.log 2>/dev/null && grep -q "=== done" /mnt/h/Research/logs/followup.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
# The retrain build (10:45) overlapped an edit of WebConfig.h (new field): objects may mix two struct layouts.
# Rebuild orbslam3_run consistently (touch -> every user of WebConfig.h recompiles). Nothing else runs at this point.
touch slam/orbslam3/include/WebConfig.h
echo "=== rebuild slam-native-teblid $(date +%T)"
cmake --build /root/build/slam-native-teblid -j8 --target orbslam3_run 2>&1 | grep -E "error|FAILED|Linking"
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; VOC=/mnt/h/Research/vocab
O=/mnt/h/Research/results/teblid_eval
export NATIVE=/root/build/slam-native-teblid/orbslam3_run
EUROC="MH_01_easy MH_03_medium MH_05_difficult V1_01_easy V1_02_medium V2_02_medium"
run() {  # name settings-variant vocab
  SPECS=()
  for e in $EUROC; do SPECS+=("euroc_$e|euroc|$D/euroc/$e|$V/EuRoC_mono_$2.yaml|$G/euroc_$e.txt|"); done
  for s in $D/tumvi/dataset-room*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$V/TUM-VI_mono_$2.yaml|$G/tumvi_$n.txt|--clahe"); done
  VOCAB=$3 FLAVOURS=native JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$1 "1 2 3" "${SPECS[@]}" > /dev/null
  echo "=== $1 done $(date +%T)"
}
echo "=== start $(date +%T)"
run orb-official stock $VOC/ORBvoc.bin
run orb-own stock $VOC/ORBvoc_own_k10L6.bin
run teblid-own teblid $VOC/TEBLIDvoc_own_k10L6.bin
P=/opt/slam/venv/bin/python
for v in orb-official orb-own teblid-own; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 200; done
$P eval/compare_builds.py --across $O/orb-own $O/teblid-own --a native --bad-ate-mm 200
$P eval/compare_builds.py --across $O/orb-official $O/orb-own --a native --bad-ate-mm 200
echo "=== done $(date +%T)"
