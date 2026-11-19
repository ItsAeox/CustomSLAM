O=/mnt/h/Research/results/klt_dev/sanity; mkdir -p $O
/root/build/slam-native/orbslam3_run --dataset euroc --mode mono --seq /mnt/h/Research/datasets/euroc/MH_01_easy --vocab /mnt/h/Research/vocab/ORBvoc.bin \
  --settings /mnt/h/Research/results/settings_variants/EuRoC_mono_klt.yaml --seed 1 --max-frames 800 --out $O > $O.log 2>&1
grep -h "Web config\|KLT-STATS" $O.log
/opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt /mnt/h/Research/results/gt/euroc_MH_01_easy.txt --est $O/traj_tum.txt --out $O/metrics.json --label sanity | tail -1
