cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; O=/mnt/h/Research/results/klt_sweep
$P eval/compare_builds.py --across $O/stock $O/klt --a ${1:-wasm} --bad-ate-mm 200
for v in stock klt; do echo "== $v (${1:-wasm})"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 200 | grep -E "sequence| ${1:-wasm} "; done
