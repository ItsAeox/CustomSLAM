#!/usr/bin/env bash
# ORB-SLAM3 port, native clang ASan+UBSan build, TUM-VI room1 seed 2 (a bad WASM seed). Not paced.
set -uo pipefail
O=/mnt/h/Research/results/scratch/asan_room1
mkdir -p $O
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:detect_stack_use_after_return=1:log_path=$O/asan
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0:log_path=$O/ubsan
/root/build/slam-native-asan/orbslam3_run --dataset tumvi --mode mono \
  --seq /mnt/h/Research/datasets/tumvi/dataset-room1_512_16 --vocab /mnt/h/Research/vocab/ORBvoc.bin \
  --settings /mnt/h/Research/CustomSLAM/baselines/orbslam3/settings/TUM-VI_mono.yaml --clahe --seed 2 \
  --out $O > $O/run.log 2>&1
echo "exit=$?"
ls -la $O
cat $O/asan.* 2>/dev/null | grep -E "ERROR|SUMMARY" | sort | uniq -c | head
cat $O/ubsan.* 2>/dev/null | grep -E "runtime error" | sed 's/^.*slam\///' | sort | uniq -c | sort -rn | head -30
