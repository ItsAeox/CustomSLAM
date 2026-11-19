#!/usr/bin/env bash
# vocab_eval2 accuracy per arm (median of per-sequence medians, Sim3 ATE mm) + runs whose wall time is off (hibernation check)
cd /mnt/h/Research/CustomSLAM; O=/mnt/h/Research/results/vocab_eval; P=/opt/slam/venv/bin/python
for a in official own-L6 own-L5 own-L4; do
  $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk -v a=$a '$2=="native"{split($0,p,"|"); split(p[3],m," "); v[++n]=m[1]} END{asort(v); printf "%-9s median-of-medians ATE %.1f mm over %d sequences\n", a, v[int((n+1)/2)], n}'
done
for f in $O/*/*/*/run_info.txt; do
  awk -v f=$f '/^frames/{n=$2} /^wall_s/{w=$2} END{}' $f
done
$P - <<'PY'
import glob,os
bad=[]
for f in glob.glob('/mnt/h/Research/results/vocab_eval/*/*/*/run_info.txt'):
    d=dict(l.split(None,1) for l in open(f) if len(l.split(None,1))==2)
    tt=[l.split() for l in open(os.path.join(os.path.dirname(f),'track_times.txt'))]
    if not tt: continue
    w=float(d.get('wall_s','0')); dur=None
    # dataset duration from the trajectory span is not stored; compare wall vs frames x nominal period
    per=0.05 if 'tumvi' in f else 0.05
    exp=int(d.get('frames','0'))*per
    if exp and (w>1.3*exp or w<0.7*exp): bad.append((round(w/exp,2),f.split('vocab_eval/')[1]))
print('runs with wall/expected outside 0.7-1.3:', len(bad)); [print(' ',b) for b in sorted(bad)[-8:]]
PY
