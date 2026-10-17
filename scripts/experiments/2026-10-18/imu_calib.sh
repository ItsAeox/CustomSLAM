#!/usr/bin/env bash
# imu_calib: does calibrating first (time offset + focal from the gyro, estimate_time_offset.py --focal) repair
# mono-inertial on phone-like inputs? imu_sens (10-16) found the two phone killers: an IMU time offset (+-20 ms: scale
# error 20-65 %, resets) and a wrong focal (+-10 %: scale collapses). The AR page measures both (web/imucalib.js);
# this checks that applying the measurement fixes mono-inertial, on TUM-VI rooms 1/3/5 (handheld), native, rt-wall,
# seeds 1-3. Phone-like = IMU stamps 40 ms late + focal assumed 10 % too long (fx, fy x 1.1).
#   base    original IMU (200 Hz), true focal                      (upper bound)
#   bad     IMU +40 ms, focal x1.1                                 (uncorrected phone)
#   cal     IMU shifted back by the estimated offset, focal x1.1 x estimated ratio (calibrated phone)
#   r60     60 Hz averaged IMU (phone rate), true focal            (upper bound at 60 Hz)
#   bad60   60 Hz + 40 ms, focal x1.1
#   cal60   60 Hz, offset and focal estimated from the 60 Hz stream
# The estimate: visual-only runs (mono, the wrong focal, seeds 1-2) per room; offset and rate ratio from each, mean of
# the two seeds. Correcting = re-sampling the original IMU with offset (40 - estimate) ms: what applying the measured
# offset in the app does.
# Usage: imu_calib.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
D=/mnt/h/Research/datasets/tumvi; S=/mnt/h/Research/datasets_synth/tumvi; G=/mnt/h/Research/results/gt
R=/mnt/h/Research/results/imu_calib
SET=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml
VOC=/mnt/h/Research/vocab/ORBvoc.bin
export NATIVE=/root/build/snap-$1/native/orbslam3_run
OFF=40; FX=1.1; ROOMS="room1 room3 room5"
fscale() { awk -v s=$1 '/^Camera1\.f[xy]:/{split($0,a,":"); printf "%s: %.6f\n", a[1], a[2]*s; next} {print}'; }
mkdir -p $R/settings $R/estimate $S
echo "=== start $(date +%T) (binary $NATIVE)"
cp $SET $R/settings/base.yaml
fscale $FX < $SET > $R/settings/bad.yaml
sed 's/^IMU.Frequency:.*/IMU.Frequency: 60.0/' $SET > $R/settings/r60.yaml
fscale $FX < $R/settings/r60.yaml > $R/settings/bad60.yaml

# 1. phone-like IMU streams
for r in $ROOMS; do n=dataset-${r}_512_16
  [ -d $S/${n}_off+$OFF ] || $P scripts/degrade_imu.py $D/$n $S/${n}_off+$OFF --rate 0 --offset-ms $OFF
  [ -d $S/${n}_r60avg ] || $P scripts/degrade_imu.py $D/$n $S/${n}_r60avg --rate 60 --average
  [ -d $S/${n}_r60avg_off+$OFF ] || $P scripts/degrade_imu.py $D/$n $S/${n}_r60avg_off+$OFF --rate 60 --average --offset-ms $OFF
done

# 2. calibrate: visual-only with the wrong focal, then offset + focal ratio against each IMU stream
for r in $ROOMS; do n=dataset-${r}_512_16
  for s in 1 2; do o=$R/estimate/$r/s$s; mkdir -p $o
    [ -s $o/traj_tum.txt ] || $NATIVE --dataset tumvi --mode mono --seq $D/$n --vocab $VOC --settings $R/settings/bad.yaml \
      --clahe --seed $s --rt-wall --out $o > $o/run.log 2>&1 &
  done
done
wait
est() {  # <traj> <imu csv> -> "offset_ms ratio"
  $P eval/estimate_time_offset.py --traj $1 --imu $2 --tbc $SET --focal 2>/dev/null | awk '
    /: offset/{for(i=1;i<=NF;i++) if($i=="offset") o=$(i+1)}
    /focal check/{for(i=1;i<=NF;i++) if($i=="x+y") f=$(i+1)} END{printf "%s %s\n", o+0, f}'
}
echo "room stream seed offset_ms ratio" > $R/estimates.txt
for r in $ROOMS; do n=dataset-${r}_512_16
  for s in 1 2; do t=$R/estimate/$r/s$s/traj_tum.txt
    echo "$r 200 $s $(est $t $S/${n}_off+$OFF/mav0/imu0/data.csv)" >> $R/estimates.txt
    echo "$r 60 $s $(est $t $S/${n}_r60avg_off+$OFF/mav0/imu0/data.csv)" >> $R/estimates.txt
  done
done
cat $R/estimates.txt
for r in $ROOMS; do n=dataset-${r}_512_16
  for hz in 200 60; do
    read -r o k <<< "$(awk -v r=$r -v h=$hz '$1==r && $2==h && $5!="nan"{o+=$4; k+=$5; c++} END{if(c) printf "%.1f %.4f", o/c, k/c; else print "nan nan"}' $R/estimates.txt)"
    [ "$o" = nan ] && { echo "  $r $hz Hz: no estimate, skipped"; continue; }
    res=$(awk -v a=$OFF -v b=$o 'BEGIN{printf "%.1f", a-b}'); fx=$(awk -v a=$FX -v b=$k 'BEGIN{printf "%.4f", a*b}')
    echo "  $r $hz Hz: offset $o ms (residual $res ms), ratio $k -> focal x$fx of true"
    if [ $hz = 200 ]; then
      rm -rf $S/${n}_cal; $P scripts/degrade_imu.py $D/$n $S/${n}_cal --rate 0 --offset-ms $res
      fscale $fx < $SET > $R/settings/cal_$r.yaml
    else
      rm -rf $S/${n}_cal60; $P scripts/degrade_imu.py $D/$n $S/${n}_cal60 --rate 60 --average --offset-ms $res
      fscale $fx < $R/settings/r60.yaml > $R/settings/cal60_$r.yaml
    fi
  done
done

# 3. mono-inertial, every condition
for c in base bad cal r60 bad60 cal60; do
  SPECS=()
  for r in $ROOMS; do n=dataset-${r}_512_16
    case $c in
      base) seq=$D/$n; st=$R/settings/base.yaml;;
      bad) seq=$S/${n}_off+$OFF; st=$R/settings/bad.yaml;;
      cal) seq=$S/${n}_cal; st=$R/settings/cal_$r.yaml;;
      r60) seq=$S/${n}_r60avg; st=$R/settings/r60.yaml;;
      bad60) seq=$S/${n}_r60avg_off+$OFF; st=$R/settings/bad60.yaml;;
      cal60) seq=$S/${n}_cal60; st=$R/settings/cal60_$r.yaml;;
    esac
    [ -f $st ] || continue
    [ -s $R/$c/tumvi_dataset-$r/native_seed3/track_times.txt ] && continue
    SPECS+=("tumvi_dataset-$r|tumvi|$seq|$st|$G/tumvi_${n}_body.txt|--clahe --mode mono_inertial")
  done
  [ ${#SPECS[@]} -gt 0 ] && FLAVOURS=native JOBS=4 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $R/$c "1 2 3" "${SPECS[@]}" > /dev/null
  echo "  $c done $(date +%T)"
done
$P scripts/experiments/2026-10-16_17/vi_table.py $R base bad cal r60 bad60 cal60
echo "=== done $(date +%T)"
