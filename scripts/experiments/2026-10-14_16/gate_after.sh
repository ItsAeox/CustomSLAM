set -uo pipefail
cmake --build /root/build/slam-native -j3 --target orbslam3_run 2>&1 | grep -E "error|FAILED" || true
S=/mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad
bash $S/gate_run.sh after &
bash $S/gate_run.sh after2 &
wait
bash $S/gate_eval.sh before before2 after after2
grep -h "Web config" /mnt/h/Research/results/gate/after/mh01.log
