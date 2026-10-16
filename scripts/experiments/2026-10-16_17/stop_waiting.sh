#!/usr/bin/env bash
# Stop a queued sweep script that is still only waiting (no runs started): kills the bash process running <name>.
# Usage: stop_waiting.sh <script name>
ps -eo pid,etime,args | grep "[b]ash .*$1" | grep -v stop_waiting | while read pid et rest; do
  if pgrep -P $pid -f orbslam3_run > /dev/null || pgrep -P $pid -f seed_sweep > /dev/null; then echo "$pid has started runs: not stopping"; continue; fi
  kill $pid && echo "stopped $pid ($rest, waited $et)"
done
