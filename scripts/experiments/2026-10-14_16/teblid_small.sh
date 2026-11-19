B=/root/build/slam-native-teblid
nice -n 19 $B/vocab_train /tmp/teblid_small.bin teblid 10 3 50 /mnt/h/Research/datasets/2011_09_26_drive_0009_sync/image_02/data > /tmp/ts.log 2>&1
echo "exit=$?"; tail -3 /tmp/ts.log
nice -n 19 $B/vocab_train /tmp/orb_small.bin orb 10 3 50 /mnt/h/Research/datasets/2011_09_26_drive_0009_sync/image_02/data > /tmp/os.log 2>&1
echo "orb exit=$?"; tail -2 /tmp/os.log
