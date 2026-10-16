#!/usr/bin/env bash
# H3 check: the runner now writes every map's trajectory; evaluate_multimap.py scores them. TUM-VI outdoors2
# (GT only at start/end; outdoors_rt: 3/6 runs unscorable with the largest-map trajectory), native, unpaced, seeds 1-2.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; n=dataset-outdoors2_512_16
O=/mnt/h/Research/results/h3_check
echo "=== start $(date +%T)"
for s in 1 2; do
  R=$O/tumvi_outdoors2/native_seed$s; mkdir -p $R
  /root/build/slam-native-teblid/orbslam3_run --dataset tumvi --mode mono --seq /mnt/h/Research/datasets/tumvi/$n \
    --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings /mnt/h/Research/results/settings_variants/TUM-VI_mono_stock.yaml \
    --out $R --seed $s --clahe > $R/run.log 2>&1 &
done
wait
for s in 1 2; do
  R=$O/tumvi_outdoors2/native_seed$s
  echo "--- seed $s: $(grep -E '^(maps|saved_maps) ' $R/run_info.txt | tr '\n' ' ') files: $(ls $R | grep -c traj_map)"
  $P eval/evaluate.py --gt /mnt/h/Research/results/gt/tumvi_$n.txt --est $R/traj_tum.txt --out $R/metrics.json 2>&1 | tail -1 | cut -c1-150
  $P eval/evaluate_multimap.py --gt /mnt/h/Research/results/gt/tumvi_$n.txt --run $R --out $R/metrics_multimap.json
  $P -c "import json; d=json.load(open('$R/metrics_multimap.json')); [print('   map', m['map'], m['est_poses'], 'poses,', m['matched_poses'], 'matched,', round(m.get('ate_sim3_rmse', float('nan'))*1000, 1), 'mm') for m in d['maps']]"
done
echo "=== done $(date +%T)"
