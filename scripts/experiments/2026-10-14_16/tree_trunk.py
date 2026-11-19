"""Headless: radial profile of the trunk part (largest brown part touching the ground) per height band."""
import bpy
import bmesh
import math

o = bpy.data.objects["Tree"]
deps = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(o.evaluated_get(deps))
bm = bmesh.new(); bm.from_mesh(me)
bm.faces.ensure_lookup_table()
seen = set(); best = None
for f in bm.faces:
    if f.index in seen:
        continue
    stack = [f]; comp = []; seen.add(f.index)
    while stack:
        g = stack.pop(); comp.append(g)
        for e in g.edges:
            for h in e.link_faces:
                if h.index not in seen:
                    seen.add(h.index); stack.append(h)
    if best is None or len(comp) > len(best):
        if min(v.co.z for g in comp for v in g.verts) < 0.0 and len(comp) > 300:
            best = comp
vs = {v for g in best for v in g.verts}
mid = [v.co for v in vs if 0.2 < v.co.z < 0.3]
cx = sum(c.x for c in mid) / len(mid); cy = sum(c.y for c in mid) / len(mid)
print("=== TRUNK axis", round(cx, 3), round(cy, 3), "verts", len(vs))
for z0 in [i * 0.025 for i in range(0, 16)]:
    band = [v.co for v in vs if z0 <= v.co.z < z0 + 0.025]
    if band:
        r = sorted(math.hypot(c.x - cx, c.y - cy) for c in band)
        print(f"z {z0:5.3f}: n {len(r):3d} r min {r[0]:.3f} med {r[len(r)//2]:.3f} max {r[-1]:.3f}")
print("=== DONE")
