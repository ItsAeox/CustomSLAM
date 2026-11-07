#!/usr/bin/env bash
# 7 Nov: page module for phone_1107 = the frozen paper sources (84c1f17, /root/src/paper-84c1f17 from
# build_paper_web_1105.sh) + one read-only binding change: WebSlam::mapStats() gets a 7th entry, the current map's id
# (atlas->GetCurrentMap()->GetId()), which orbslam3_run's trace writes and the scorer needs for "largest map". The
# frozen module (web/build, md5 c4608c77...) has 6 entries. No SLAM source changes.
# Steps: patch bindings.cc in the paper export, incremental build (only bindings.cc + link), copy to web/build-phone/,
# revert the patch, rebuild, and check that the module is byte-identical to the frozen one again (md5 c4608c77...).
set -uo pipefail
LOG=/mnt/h/Research/results/phone_1107/build_phone_web_1107.log
S=/root/src/paper-84c1f17 B=/root/build/slam-wasm-paper W=/mnt/h/Research/CustomSLAM/web
F=$S/slam/port/web/bindings.cc
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
{
  echo "=== start $(date +%T)"
  cp $F /root/src/bindings_84c1f17.cc.orig
  python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = "        stats_[5] = map->GetIniertialBA2() ? 1 : 0;\n      }"
b = "        stats_[5] = map->GetIniertialBA2() ? 1 : 0;\n        stats_[6] = (int)map->GetId();  // phone_1107: current map id (read only)\n      }"
assert s.count(a) == 1; s = s.replace(a, b)
a = "    stats_[0] = stats_[1] = stats_[3] = stats_[4] = stats_[5] = 0;"
assert s.count(a) == 1; s = s.replace(a, a + "\n    stats_[6] = -1;")
a = "    return val(typed_memory_view(6, stats_));"
assert s.count(a) == 1; s = s.replace(a, "    return val(typed_memory_view(7, stats_));")
a = "  int stats_[6] = {0, 0, 0, 0, 0, 0};"
assert s.count(a) == 1; s = s.replace(a, "  int stats_[7] = {0, 0, 0, 0, 0, 0, -1};")
open(p, "w").write(s)
print("patched")
EOF
  diff /root/src/bindings_84c1f17.cc.orig $F
  nice -n 10 cmake --build $B -j6 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking|Building"
  if [ $B/orbslam3.wasm -nt $F ]; then
    mkdir -p $W/build-phone && cp $B/orbslam3.js $B/orbslam3.wasm $B/orbslam3.worker.mjs $W/build-phone/
    md5sum $W/build-phone/orbslam3.wasm
  else
    echo "BUILD FAILED"
  fi
  cp /root/src/bindings_84c1f17.cc.orig $F && touch $F
  nice -n 10 cmake --build $B -j6 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
  md5sum $B/orbslam3.wasm $W/build/orbslam3.wasm
  echo "=== done $(date +%T)"
} >> $LOG 2>&1
