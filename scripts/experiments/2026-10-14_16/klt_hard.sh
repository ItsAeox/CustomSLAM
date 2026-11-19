S=/mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad
for v in klt stock; do
  bash $S/klt_test.sh $v $v euroc/MH_05_difficult euroc/V2_02_medium tumvi/dataset-room1_512_16 tumvi/dataset-room2_512_16
done
