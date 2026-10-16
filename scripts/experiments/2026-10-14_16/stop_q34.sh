#!/usr/bin/env bash
# Stop queue v3 (parent first, so it cannot move on), its sweeps and runners, then the waiting queue v4.
for p in $(ps -eo pid,args | grep '[b]ash .*run_queue3\.sh' | cut -c1-8); do kill "$p"; done
sleep 1
for p in $(ps -eo pid,args | grep '[s]eed_sweep\.sh' | cut -c1-8); do kill "$p"; done
for p in $(ps -eo pid,args | grep -E '[o]rbslam3_run' | cut -c1-8); do kill "$p"; done
for p in $(ps -eo pid,args | grep '[b]ash .*run_queue4\.sh' | cut -c1-8); do kill "$p"; done
sleep 2
echo "remaining:"; ps -eo pid,args | grep -E '[r]un_queue|[s]eed_sweep|[o]rbslam3_run' | cut -c1-100
free -m | sed -n 2p
