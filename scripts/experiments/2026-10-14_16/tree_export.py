"""Headless: trim the tree's base (stones, moss, roots, mushrooms), cut the trunk flat, export web/assets/tree.glb."""
import bpy
import bmesh
from mathutils import Vector

ZCUT = 0.15   # trunk is a plain cylinder above this (roots flare below)
OUT = "H:/Research/CustomSLAM/web/assets/tree.glb"

src = bpy.data.objects["Tree"]
mat = src.data.materials[0]
print("=== material nodes", [(n.type, n.name) for n in mat.node_tree.nodes] if mat.use_nodes else "no nodes")
deps = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(src.evaluated_get(deps))  # geometry nodes applied

bm = bmesh.new(); bm.from_mesh(me)
bm.faces.ensure_lookup_table()
# 1. delete loose parts that lie entirely at the base (stones, moss patch, root bits, mushrooms)
seen = set(); drop = []
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
    if max(v.co.z for g in comp for v in g.verts) < 0.2:
        drop.extend(comp)
print("=== dropping base parts:", len(drop), "faces")
bmesh.ops.delete(bm, geom=list({f for f in drop}), context="FACES")
# 2. cut the trunk flat at ZCUT (removes the root flare), cap the hole
geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, ZCUT), plane_no=(0, 0, 1), clear_inner=True)
cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
# 3. base on the floor (z = 0), centred on the trunk axis
zmin = min(v.co.z for v in bm.verts)
base = [v.co for v in bm.verts if v.co.z < zmin + 0.01]
cx = sum(c.x for c in base) / len(base); cy = sum(c.y for c in base) / len(base)
bmesh.ops.translate(bm, verts=bm.verts[:], vec=(-cx, -cy, -zmin))
bm.normal_update()
me2 = bpy.data.meshes.new("TreeTrimmed"); bm.to_mesh(me2); bm.free()
me2.materials.append(mat)
obj = bpy.data.objects.new("TreeTrimmed", me2)
bpy.context.scene.collection.objects.link(obj)
zs = [v.co.z for v in me2.vertices]
print(f"=== trimmed: verts {len(me2.vertices)} faces {len(me2.polygons)} height {max(zs):.3f}")

for o in bpy.data.objects:
    o.select_set(o is obj)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_yup=True,
                          export_apply=True, export_materials="EXPORT")
print("=== exported", OUT)

# preview
scene = bpy.context.scene
src.hide_render = True
for o in bpy.data.objects:
    if o.type == "MESH" and o is not obj:
        o.hide_render = True
cam = scene.camera
if cam is None:
    cd = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cd); scene.collection.objects.link(cam); scene.camera = cam
h = max(zs)
cam.location = Vector((0, -2.4 * h, 0.35 * h)); cam.rotation_euler = (1.5708 - 0.1, 0, 0)
engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in engines else scene.render.engine
try:
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.light = "STUDIO"
except Exception:
    pass
scene.render.resolution_x = 600; scene.render.resolution_y = 800
scene.render.filepath = "C:/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/tree_trimmed.png"
bpy.ops.render.render(write_still=True)
print("=== DONE")
