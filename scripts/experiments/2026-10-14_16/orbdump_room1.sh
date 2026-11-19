#!/usr/bin/env bash
# CLAHE + ORB extraction checksums, native vs WASM, on room1 frames
source /opt/slam/emsdk/emsdk_env.sh >/dev/null 2>&1
I=/mnt/h/Research/datasets/tumvi/dataset-room1_512_16/mav0/cam0/data
for f in 1520530308199447626 1520530333150172890 1520530383152254609 1520530433153693398; do
  /root/build/slam-native/orb_dump $I/$f.png 1 1500 > /tmp/n.txt; cp /tmp/n.txt /mnt/h/Research/results/scratch/orbdump_n_$f.txt 2>&1
  (cd $I && node /root/build/slam-wasm/orb_dump.js $f.png 1 1500 > /tmp/w.txt 2>&1); cp /tmp/w.txt /mnt/h/Research/results/scratch/orbdump_w_$f.txt
  echo "== $f: $(diff /tmp/n.txt /tmp/w.txt | grep -c '^<') differing lines of $(wc -l < /tmp/n.txt)"
  diff /tmp/n.txt /tmp/w.txt | grep -E "ORB "
done
