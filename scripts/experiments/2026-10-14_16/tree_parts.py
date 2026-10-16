"""Headless: apply the tree's modifiers, split into loose parts, describe each (height range, extent, mean texture colour)."""
import bpy
import bmesh

o = bpy.data.objects["Tree"]
deps = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(o.evaluated_get(deps))
bm = bmesh.new(); bm.from_mesh(me)
uv = bm.loops.layers.uv.active
img = bpy.data.images["tree_bace color.png"]
W, H = img.size
px = list(img.pixels)  # RGBA float, row-major from bottom


def colour(u, v):
    x = min(W - 1, max(0, int((u % 1.0) * W))); y = min(H - 1, max(0, int((v % 1.0) * H)))
    i = 4 * (y * W + x)
    return px[i], px[i + 1], px[i + 2]


bm.faces.ensure_lookup_table()
seen = set(); parts = []
for f in bm.faces:
    if f.index in seen:
        continue
    stack = [f]; comp = []
    seen.add(f.index)
    while stack:
        g = stack.pop(); comp.append(g)
        for e in g.edges:
            for h in e.link_faces:
                if h.index not in seen:
                    seen.add(h.index); stack.append(h)
    vs = {v for g in comp for v in g.verts}
    zs = [v.co.z for v in vs]; xs = [v.co.x for v in vs]; ys = [v.co.y for v in vs]
    cs = [colour(*l[uv].uv) for g in comp for l in g.loops] if uv else [(0, 0, 0)]
    mc = [sum(c[i] for c in cs) / len(cs) for i in range(3)]
    parts.append((min(zs), max(zs), max(xs) - min(xs), max(ys) - min(ys), len(comp), mc))
parts.sort(key=lambda p: p[0])
print("=== PARTS", len(parts), "faces", len(bm.faces))
for i, (z0, z1, dx, dy, nf, mc) in enumerate(parts):
    print(f"part {i:3d} z {z0:6.3f}..{z1:6.3f} extent {dx:5.3f}x{dy:5.3f} faces {nf:4d} colour {mc[0]:.2f},{mc[1]:.2f},{mc[2]:.2f}")
print("=== DONE")
