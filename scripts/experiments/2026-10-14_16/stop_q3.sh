#!/usr/bin/env bash
# Stop the waiting queue v3 script (it has not started anything yet).
for p in $(ps -eo pid,args | grep '[b]ash .*run_queue3\.sh' | cut -c1-8); do kill "$p"; done
sleep 1
ps -eo pid,args | grep '[r]un_queue3' | cut -c1-100
echo "stopped"
