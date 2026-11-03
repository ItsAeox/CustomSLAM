#!/usr/bin/env bash
# gyrobias_1103: why page-betagyro / page-mapfirst fail on EuRoC in main_1103 (46 / 53 of 55 bad) but not on TUM-VI
# (0 / 30). betagyro = beta + online metric scale (Web.Scale.*) + gyro rotation in the pose solve (Web.Gyro.PoseSigma
# 0.05 deg, PoseTau 5 s attitude reference). Hypothesis: EuRoC's raw gyro bias (4.6-5.0 deg/s static-start mean;
# the synthetic phone model assumes 0.11 deg/s) pulls the attitude reference off. Arms, same frozen build snap-84c1f17:
#   beta-raw / betagyro-raw          main_1103 settings, the dataset IMU (repeat of main_1103 under this batch's load)
#   scale-raw / pose-raw             beta + only the scale lines / only the pose-prior lines (attribution)
#   beta-cal / pose-cal / betagyro-cal   the same settings on a copy whose gyro has the dataset's estimated bias removed
#                                    (gyro_debias.py; an oracle calibrated gyroscope, diagnostic only)
# MH01, MH04, V1_01, V2_02, seeds 1-2, rt-wall, 4 at a time. Waits for ext_1103 to finish (no overlap with a batch).
set -uo pipefail
R=/mnt/h/Research; REPO=$R/CustomSLAM; O=$R/results/gyrobias_1103; P=/opt/slam/venv/bin/python
X=$REPO/scripts/experiments/2026-11-03; CAL=/root/data/euroc_gyrocal
until grep -q "^=== done" $R/logs/ext_1103.log 2>/dev/null; do sleep 60; done
mkdir -p $O/settings
M=$R/results/main_1103/settings
SC=$'Web.Scale.Online: 1\nWeb.Scale.Method: 1'; PO=$'Web.Gyro.PoseSigma: 0.05\nWeb.Gyro.PoseTau: 5'
cp $M/EuRoC_mono_page-beta.yaml $O/settings/beta.yaml
cp $M/EuRoC_mono_page-betagyro.yaml $O/settings/betagyro.yaml
{ cat $M/EuRoC_mono_page-beta.yaml; printf '\n# gyrobias_1103: scale only\n%s\n' "$SC"; } > $O/settings/scale.yaml
{ cat $M/EuRoC_mono_page-beta.yaml; printf '\n# gyrobias_1103: pose prior only\n%s\n' "$PO"; } > $O/settings/pose.yaml
SEQS="MH_01_easy MH_04_difficult V1_01_easy V2_02_medium"
for s in $SEQS; do $P $X/gyro_debias.py --seq $R/datasets/euroc/$s --out $CAL/$s; done
sha256sum $O/settings/*.yaml > $O/BUILD.txt
for arm in beta-raw betagyro-raw scale-raw pose-raw beta-cal pose-cal betagyro-cal; do
  for s in $SEQS; do for k in 1 2; do echo "$arm $s $k"; done; done
done | $P -c "import sys,random; l=sys.stdin.readlines(); random.Random(11033).shuffle(l); print(''.join(l), end='')" > $O/runlist.txt
echo "=== start $(date +%T): $(wc -l < $O/runlist.txt) runs"
xargs -P 4 -L 1 bash $X/gyrobias_job.sh < $O/runlist.txt
echo "=== runs done $(date +%T)"
$P $X/score_main_1103.py --root $O --load /nonexistent --title "gyrobias_1103 results (diagnostic)" \
  --arms beta-raw betagyro-raw scale-raw pose-raw beta-cal pose-cal betagyro-cal > /dev/null && echo scored
echo "=== done $(date +%T)"
