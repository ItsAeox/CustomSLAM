#!/usr/bin/env bash
# Syntax-only compile of changed ORB-SLAM3 sources with the native build's exact flags (no objects written).
# Usage: syntax_check.sh <file.cc> ...   (paths relative to slam/orbslam3/src)
set -uo pipefail
B=/root/build/slam-native-teblid
for f in "$@"; do
  cmd=$(cd $B && ninja -t commands orbslam3_run 2>/dev/null | grep -m1 "/src/$f")
  [ -z "$cmd" ] && { echo "$f: no compile command found"; continue; }
  cmd=$(echo "$cmd" | sed 's/ -o [^ ]*\.o / -fsyntax-only /; s/ -MD -MT [^ ]* -MF [^ ]*//')
  (cd $B && eval "nice -n 10 $cmd") && echo "$f: OK" || echo "$f: FAILED"
done
