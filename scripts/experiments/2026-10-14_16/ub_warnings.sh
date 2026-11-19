#!/usr/bin/env bash
# clang -fsyntax-only with UB-relevant warnings over ORB-SLAM3's sources (same include dirs/defines as the build)
cd /root/build/slam-native-asan
python3 - <<'PY' > /tmp/cmds.sh
import json
W = "-Wreturn-type -Wuninitialized -Wsometimes-uninitialized -Wconditional-uninitialized -Wreturn-stack-address " \
    "-Wdangling -Wdangling-gsl -Wself-assign -Wsizeof-pointer-memaccess -Wint-in-bool-context"
PAT = r"warning:.*\[-W(return-type|uninitialized|sometimes-uninitialized|conditional-uninitialized|return-stack-address|dangling|dangling-gsl|self-assign|sizeof-pointer-memaccess|int-in-bool-context)\]"
for e in json.load(open("compile_commands.json")):
    f = e["file"]
    if "/orbslam3/src/" not in f and "/port/" not in f:
        continue
    c = e["command"].split(" -o ")[0] + " " + f
    print(f"cd {e['directory']} && {c} -fsyntax-only {W} 2>&1 | grep -E '{PAT}'")
PY
bash /tmp/cmds.sh | sort | uniq | grep -v "Thirdparty/Sophus\|/usr/include" > /tmp/ub_warnings.txt
wc -l < /tmp/ub_warnings.txt
sed 's#/mnt/h/Research/CustomSLAM/slam/##' /tmp/ub_warnings.txt | cut -c1-220
