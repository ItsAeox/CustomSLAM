#!/usr/bin/env bash
# Stop the queued TEBLID evaluation (its -j8 rebuild would exhaust WSL's memory) and any build it started.
ps -eo pid,etime,args | grep -E '[r]un_teblid_eval3|[c]make --build|[n]inja |[c]c1plus' | cut -c1-120
for p in $(ps -eo pid,args | grep '[r]un_teblid_eval3\.sh' | cut -c1-8); do kill "$p"; done
for p in $(ps -eo pid,args | grep '[c]make --build /root/build/slam-native-teblid' | cut -c1-8); do kill "$p"; done
for p in $(ps -eo pid,comm | grep -E ' (ninja|cc1plus)$' | cut -c1-8); do kill "$p"; done
sleep 3
echo "after:"
ps -eo pid,args | grep -E '[r]un_teblid_eval3|[n]inja |[c]c1plus|[o]rbslam3_run' | cut -c1-120
free -m | sed -n 2p
echo "teblid_eval.log:"; tail -3 /mnt/h/Research/logs/teblid_eval.log
