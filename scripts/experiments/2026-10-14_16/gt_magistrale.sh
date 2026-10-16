cd /mnt/h/Research/CustomSLAM
for i in 1 2 3 4 5 6; do
  n=dataset-magistrale${i}_512_16
  nice -n 19 /opt/slam/venv/bin/python eval/gt_asl.py --dataset tumvi --seq /mnt/h/Research/datasets/tumvi/$n \
    --out /mnt/h/Research/results/gt/tumvi_$n.txt 2>&1 | tail -2
done
