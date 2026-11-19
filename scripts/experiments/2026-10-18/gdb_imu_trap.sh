#!/usr/bin/env bash
# G24: where does an IMU quantity first turn non-finite? Native build with IMU::WebNanTrap (called from the bias /
# velocity / pose setters and Preintegrated::GetDeltaRotation), deterministic unpaced repro (rec 110106, seed 1,
# mono-inertial, -88 ms): gdb stops at the first few traps with a backtrace each, then the abort.
# Usage: gdb_imu_trap.sh <out dir> [recording] [seed]
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
O=$1 R=${2:-rec_20261018_110106} S=${3:-1}
B=${B:-/root/build/slam-native-teblid/orbslam3_run} PH=/mnt/h/Research/datasets/phone
ST=/mnt/h/Research/results/replays_1018_imu/$R/settings/${R}_arm-imu_dt88.yaml
mkdir -p "$O"
cat > "$O/cmds.gdb" <<'EOF'
set pagination off
handle SIGPIPE nostop noprint
break ORB_SLAM3::IMU::WebNanTrap
commands
  silent
  bt 14
  continue
end
run
bt 30
EOF
gdb -q -batch -x "$O/cmds.gdb" --args $B --dataset euroc --mode mono_inertial --seq $PH/$R --vocab /mnt/h/Research/vocab/ORBvoc.bin \
  --settings $ST --seed $S --imu-dt -88 --out $O > $O/gdb.log 2>&1
echo "exit $?; traps $(grep -c 'IMU-NAN first seen' $O/gdb.log); aborts $(grep -c 'SO3::exp failed' $O/gdb.log)"
grep -m 12 "IMU-NAN" $O/gdb.log
