#!/usr/bin/env bash
# Per-sequence side-by-side table for kf_policy (median ATE mm and bad count per arm)
cd /mnt/h/Research/CustomSLAM; O=/mnt/h/Research/results/kf_policy; P=/opt/slam/venv/bin/python
for a in stock kf080 kf070 cull080 cull070; do
  $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk -v a=$a '$2=="native"{split($0,p,"|"); split(p[3],m," "); print a, $1, $5, m[1]}'
done > /tmp/kf_rows.txt
$P - <<'PY'
rows=[l.split() for l in open('/tmp/kf_rows.txt')]
arms=['stock','kf080','kf070','cull080','cull070']; d={}
for a,s,b,m in rows: d.setdefault(s,{})[a]=(b,m)
print('%-24s'%'seq'+''.join('%14s'%a for a in arms))
for s in sorted(d): print('%-24s'%s.replace('tumvi_dataset-','').replace('euroc_','')+''.join('%14s'%('%s/%s'%(d[s].get(a,('-','-'))[1],d[s].get(a,('-','-'))[0])) for a in arms))
PY
