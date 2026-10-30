#!/usr/bin/env bash
# slam/eqvio_web against eqvio_opt (eqvio_checks.sh) on a few synthetic sequences (phone IMU) and the real 1 m slides:
#   native_color  eqvio_run --color (BGR frames as eqvio_opt: the same filter and tracker -> should match eqvio_opt)
#   native_gray   eqvio_run (grey frames, as the browser gives them)
#   wasm_gray     the same under Node (WebAssembly, SIMD128): the browser build's accuracy and cost
# One run at a time (timing.csv: GIFT ms, filter ms per frame). Scores with the WSL venv (evo).
#   eqvio_web_parity.sh ["<seq ...>"]
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
SEQS=${1:-"ar2 rotate2 slide3 orbit2 rec_20261029_202938"}
O=${O:-/mnt/h/Research/results/eqvio_web_1030}; C=/mnt/h/Research/CustomSLAM/baselines/eqvio/config_synth_phone.yaml
EV=/mnt/h/Research/CustomSLAM/eval; PY=/opt/slam/venv/bin/python
for s in $SEQS; do
  d=/mnt/h/Research/datasets/synth/$s; [[ $s == rec_* ]] && d=/mnt/h/Research/datasets/phone/$s
  for v in native_color native_gray wasm_gray; do
    o=$O/$s/$v; mkdir -p $o
    if [ ! -s $o/timing.csv ]; then
      t0=$(date +%s.%N)
      case $v in
        native_color) /root/build/eqvio-native/eqvio_run $d $C $o --color ;;
        native_gray) /root/build/eqvio-native/eqvio_run $d $C $o ;;
        wasm_gray) node /root/build/eqvio-wasm/eqvio_run.js $d $C $o ;;
      esac > $o/run.log 2>&1
      echo "$s $v: exit $? in $(echo "$(date +%s.%N) $t0" | awk '{printf "%.1f", $1 - $2}') s"
    fi
    if [[ $s == rec_* ]]; then
      r=$($PY $EV/slides_metric.py $o/traj_camclock.txt | sed 's/^[^:]*: //')
    else
      $PY $EV/evaluate.py --gt /mnt/h/Research/results/gt/synth_$s.txt --est $o/traj_camclock.txt --out $o/metrics.json > /dev/null 2>&1
      r=$($PY -c "import json; m = json.load(open('$o/metrics.json')); print(f\"SE3 ATE {100 * m['ate_se3']['rmse']:.1f} cm, scale {m['sim3_scale']:.3f}, Sim3 ATE {100 * m['ate_sim3']['rmse']:.1f} cm, {m['matched_poses']} poses\")" 2>&1 | tail -1)
    fi
    tm=$($PY -c "import numpy as np; d = np.genfromtxt('$o/timing.csv', delimiter=',', names=True); print(' / '.join(f'{k} {np.median(d[k]):.2f} (p90 {np.percentile(d[k], 90):.2f})' for k in ('features_ms', 'filter_ms', 'frame_ms')))")
    echo "$s $v: $r | ms median: $tm"
  done
done
