#!/usr/bin/env bash
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; O=/mnt/h/Research/results/m3_variants
for v in stock light-map light-orb light-all; do echo "=== $v"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 200; done
for v in light-map light-orb light-all; do echo "=== stock (A) vs $v (B), wasm"; $P eval/compare_builds.py --across $O/stock $O/$v --a wasm --bad-ate-mm 200 | tail -14; done
# mapping throughput: KFs inserted vs local BAs done, and mean local-mapping time per KF (REGISTER_TIMES)
for v in stock light-map light-orb light-all; do
  python3 - "$O/$v" "$v" <<'PY'
import sys, re, glob, statistics as st
d, v = sys.argv[1], sys.argv[2]
kf, lba, lm, orb = [], [], [], []
for log in glob.glob(f"{d}/*/wasm_seed*/run.log"):
    t = open(log, errors="replace").read()
    m = re.search(r"LM-STATS keyframes=(\d+) lba_done=(\d+)", t)
    if m: kf.append(int(m.group(1))); lba.append(int(m.group(2)))
    run = log.rsplit("/", 1)[0]
    try:
        e = open(f"{run}/ExecMean.txt").read()
        a = re.search(r"Total Local Mapping: ([\d.]+)", e); b = re.search(r"ORB Extraction: ([\d.]+)", e)
        if a: lm.append(float(a.group(1)))
        if b: orb.append(float(b.group(1)))
    except OSError:
        pass
print(f"{v:10s} runs {len(kf)} | KFs/run {st.mean(kf):.0f} | LBA done/run {st.mean(lba):.0f} | "
      f"local mapping ms/KF {st.mean(lm):.0f} | ORB extraction ms {st.mean(orb):.1f}")
PY
done
