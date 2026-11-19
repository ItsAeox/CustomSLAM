#!/usr/bin/env bash
# WSL memory and the SLAM runner processes (RSS), for keeping concurrent runs within 7.8 GB
free -m | head -2
ps -eo rss,args --sort=-rss | grep '[o]rbslam3_run' | awk '{printf "%5d MB  %s %s\n", $1/1024, $2, $NF}' | head -10
ps -eo etime,args | grep -E '[c]c1plus|[c]lang\+\+|[c]lang -cc1' | grep -o '[A-Za-z0-9_]*\.cc\b\|[A-Za-z0-9_]*\.cpp\b' | head -3 | sed 's/^/compiling: /'
