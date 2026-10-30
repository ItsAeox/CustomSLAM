#!/usr/bin/env bash
# EqVIO's initial landmark depth (eqf:initialValue:sceneDepth; EuRoC-tuned config: 5.0 m) on the real phone recordings and
# the synthetic set (phone IMU, variants 0-5). With 5 m EqVIO diverged (km) on every real recording; with 2 m the real 1 m
# slides came out at 0.966 m (scratch test 30 Oct). This sweep: does the prior decide the result (sensitivity)?
# Uses eqvio_checks.sh's folders on the WSL disk (/root/eqvio_data/<run>: sensor.yaml, placeholder ground truth for the
# recordings). Configs: baselines/eqvio/config_synth_phone.yaml with sceneDepth replaced -> <out>/config_d<depth>.yaml.
# Scoring: eqvio_checks_score.py <out>/d<depth>.
#   eqvio_depth_sweep.sh "<depths>" [parallel jobs, default 8]
set -uo pipefail
DEPTHS=${1:-"1 1.5 2 3"}; P=${2:-8}
export O=/mnt/h/Research/results/eqvio_depth_1030
mkdir -p $O
for dep in $DEPTHS; do
  sed -E "s/^(    sceneDepth: ).*/\1$dep/" /mnt/h/Research/CustomSLAM/baselines/eqvio/config_synth_phone.yaml | sed 's/\r$//' > $O/config_d$dep.yaml
done
one() {  # <depth> <run name>
  local o=$O/d$1/$2
  [ -s $o/IMUState.csv ] && return
  mkdir -p $o
  ( cd $o && timeout 1800 /root/build/eqvio/eqvio_opt /root/eqvio_data/$2/ $O/config_d$1.yaml -o $o/ --mode asl --timing > $o/run.log 2>&1 )
  echo "d$1 $2: exit $?"
}
export -f one
{
  for dep in $DEPTHS; do
    for r in rec_20261029_202938 rec_20261021_070221 rec_20261018_105829 rec_20261018_105951 rec_20261018_110106 \
             rec_20261018_110210 rec_20261018_110310 rec_20261018_110407 rec_20261018_110500 rec_20261018_110548 \
             rec_20261018_110632 rec_20261018_110751 rec_20261018_184922 rec_20261018_185037 rec_20261018_185134 \
             rec_20261018_220132 rec_20261018_220403; do echo "$dep $r"; done
    for sc in ar gesture orbit rotate shake slide walk; do for v in 0 1 2 3 4 5; do echo "$dep $sc$v"; done; done
  done
} | xargs -P $P -L 1 bash -c 'one $0 $1'
echo "=== all done $(date +%T)"
