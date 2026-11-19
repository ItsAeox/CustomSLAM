# evaluate gate runs: tags...
for t in "$@"; do
  for s in mh01 room1; do
    gt=/mnt/h/Research/results/gt/euroc_MH_01_easy.txt; [ $s = room1 ] && gt=/mnt/h/Research/results/gt/tumvi_dataset-room1_512_16.txt
    d=/mnt/h/Research/results/gate/$t/$s
    [ -f $d/traj_tum.txt ] && /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt $gt --est $d/traj_tum.txt --out $d/metrics.json --label "$t $s" | tail -1
  done
done
