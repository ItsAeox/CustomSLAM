cd /root/build/slam-native && ninja -t compdb > /tmp/cc.json 2>/dev/null
CMD=$(python3 -c "
import json
for e in json.load(open('/tmp/cc.json')):
    if e['file'].endswith('src/ORBextractor.cc'):
        print(e['command'].split(' -o ')[0]); break")
P=/mnt/h/Research/CustomSLAM/slam/port/features/teblid
for f in /mnt/h/Research/CustomSLAM/slam/orbslam3/src/ORBextractor.cc $P/beblid.cpp /mnt/h/Research/CustomSLAM/slam/port/tools/vocab_train.cc /mnt/h/Research/CustomSLAM/slam/orbslam3/src/Tracking.cc; do
  nice -n 19 bash -c "$CMD -I$P -fsyntax-only $f" 2>&1 | grep -E "error" | head -5; echo "checked $(basename $f)"
done
