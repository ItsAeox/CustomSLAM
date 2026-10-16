#!/usr/bin/env bash
cd /mnt/h/Research/datasets/euroc/V2_02_medium/mav0/cam0/data
ls | sed -n 1000,1099p > /tmp/l.txt
s=$(date +%s%N); while read f; do cat "$f" > /dev/null; done < /tmp/l.txt; e=$(date +%s%N)
echo "cold: $(( (e - s) / 100000000 )) ms per PNG (x0.1 precision: $(( (e - s) / 10000000 )) /10)"
s=$(date +%s%N); while read f; do cat "$f" > /dev/null; done < /tmp/l.txt; e=$(date +%s%N)
echo "warm: $(( (e - s) / 10000000 )) /10 ms per PNG"
