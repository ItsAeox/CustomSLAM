"""Headless: export the whole tree (roots, stones, moss kept) with its lowest point at z = 0, centred on the trunk."""
import bpy
import bmesh

OUT = "H:/Research/CustomSLAM/web/assets/tree.glb"
src = bpy.data.objects["Tree"]
mat = src.data.materials[0]
deps = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(src.evaluated_get(deps))  # geometry nodes applied
bm = bmesh.new(); bm.from_mesh(me)
zmin = min(v.co.z for v in bm.verts)
TRUNK = (0.012, 0.035)  # trunk axis (measured: mean of trunk vertices at z 0.2..0.3)
bmesh.ops.translate(bm, verts=bm.verts[:], vec=(-TRUNK[0], -TRUNK[1], -zmin))
me2 = bpy.data.meshes.new("TreeFull"); bm.to_mesh(me2); bm.free()
me2.materials.append(mat)
obj = bpy.data.objects.new("TreeFull", me2)
bpy.context.scene.collection.objects.link(obj)
for o in bpy.data.objects:
    o.select_set(o is obj)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_yup=True,
                          export_apply=True, export_materials="EXPORT")
zs = [v.co.z for v in me2.vertices]
print(f"=== exported full tree: verts {len(me2.vertices)} height {max(zs):.3f} -> {OUT}")
