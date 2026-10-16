ps -eo pid,etime,pcpu,rss,args | grep vocab_train | grep -v grep | cut -c1-120
dmesg 2>/dev/null | grep -i "killed process" | tail -3
