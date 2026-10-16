P=/opt/slam/venv/bin/python; S=/mnt/h/Research/results/scratch/rot_align_check.py; G=/mnt/h/Research/results/gt
for s in MH_01_easy MH_03_medium V1_01_easy V1_02_medium V2_02_medium; do
  echo "$s: $($P $S $G/euroc_$s.txt /mnt/h/Research/results/m3_variants/light-map/euroc_$s/wasm_seed1/traj_tum.txt 2>&1 | tail -1)"
done
