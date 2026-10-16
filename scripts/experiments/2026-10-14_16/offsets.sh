P=/opt/slam/venv/bin/python; S=/mnt/h/Research/results/scratch/rot_over_time.py; G=/mnt/h/Research/results/gt
for s in MH_01_easy MH_03_medium MH_05_difficult V1_01_easy V1_02_medium V2_02_medium; do
  f=/mnt/h/Research/results/m3_variants/light-map/euroc_$s/wasm_seed1/traj_tum.txt
  echo "$s: $($P $S $G/euroc_$s.txt $f | tail -1)"
done
for s in room2 room4 room6; do
  f=/mnt/h/Research/results/m3_variants/light-map/tumvi_dataset-$s/wasm_seed1/traj_tum.txt
  echo "$s: $($P $S $G/tumvi_dataset-${s}_512_16.txt $f | tail -1)"
done
