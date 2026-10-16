#!/usr/bin/env bash
# Stop the waiting (not started) queue scripts: gates, magistrale, hard10 -- replaced by --rt-wall versions.
for s in run_gates.sh run_magistrale.sh run_hard10.sh; do
  for p in $(ps -eo pid,args | grep "[b]ash .*$s" | cut -c1-8); do kill "$p"; done
done
sleep 1
ps -eo pid,args | grep -E '[r]un_(gates|magistrale|hard10)|[o]rbslam3_run' | cut -c1-120
echo "remaining shown above (should be none)"
for f in klt_gates magistrale hard10; do echo "$f.log: $(tail -1 /mnt/h/Research/logs/$f.log 2>/dev/null)"; done
