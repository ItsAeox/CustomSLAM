#!/usr/bin/env bash
# sanity: compile Tracking.cc with the scan flags and show the raw diagnostics summary
L=$(grep -m1 "Tracking.cc" /tmp/cmds.sh | sed 's/ 2>&1 | grep.*//')
bash -c "$L" > /tmp/one.out 2>&1; echo "exit=$?"
grep -c "warning:" /tmp/one.out; grep -c "error:" /tmp/one.out; tail -3 /tmp/one.out | cut -c1-200
# whole-tree: all warnings of any kind from ORB-SLAM3 sources (not headers), counted by flag
for l in $(seq 1 $(wc -l < /tmp/cmds.sh)); do
  C=$(sed -n "${l}p" /tmp/cmds.sh | sed 's/ 2>&1 | grep.*//')
  bash -c "$C -Wall -Wextra" 2>&1 | grep "warning:" | grep "/orbslam3/src/\|/port/"
done | grep -o "\[-W[a-zA-Z0-9-]*\]" | sort | uniq -c | sort -rn
