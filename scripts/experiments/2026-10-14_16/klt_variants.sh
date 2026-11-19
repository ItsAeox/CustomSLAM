bash /mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/build_wasm_only.sh > /dev/null
O=/mnt/h/Research/results/klt_dev/wasm
VARS="klt klt-w15l2 klt-w11l2 klt-w15l2nofb klt-w15l2i10nofb" bash /mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/klt_wasm.sh
for v in klt klt-w15l2 klt-w11l2 klt-w15l2nofb klt-w15l2i10nofb; do
  /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt /mnt/h/Research/results/gt/euroc_MH_01_easy.txt --est $O/$v/traj_tum.txt --out $O/$v/metrics.json --label "$v" | tail -1
done
