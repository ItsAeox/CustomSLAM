# ORB descriptor checksums (WASM orb_dump) on a few frames -> $1
source /opt/slam/emsdk/emsdk_env.sh >/dev/null 2>&1
I=/mnt/h/Research/datasets/tumvi/dataset-room1_512_16/mav0/cam0/data; E=/mnt/h/Research/datasets/euroc/MH_01_easy/mav0/cam0/data
: > $1
for f in $I/1520530308199447626.png $I/1520530383152254609.png $E/$(ls $E | sed -n 1500p); do
  (cd $(dirname $f) && node /root/build/slam-wasm/orb_dump.js $(basename $f) 1 1500 2>&1 | grep "ORB " >> $1)
done
cat $1
