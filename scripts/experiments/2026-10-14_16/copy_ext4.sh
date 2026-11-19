#!/usr/bin/env bash
# Copy cam0 (images + data.csv) of EuRoC and the TUM-VI rooms to WSL ext4 (fast I/O for --live / honest real time).
set -uo pipefail
SRC=/mnt/h/Research/datasets; DST=/root/datasets
for s in $SRC/euroc/* $SRC/tumvi/dataset-room*; do
  rel=${s#$SRC/}
  mkdir -p $DST/$rel/mav0/cam0
  nice -n 19 ionice -c3 cp -r $s/mav0/cam0/data $s/mav0/cam0/data.csv $DST/$rel/mav0/cam0/
  echo "$rel $(ls $DST/$rel/mav0/cam0/data | wc -l) images $(date +%T)"
done
du -sh $DST
echo "=== copy done $(date +%T)"
