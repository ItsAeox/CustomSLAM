"""Headless: describe the tree .blend (objects, meshes, bounds, materials) and render a preview."""
import bpy
from mathutils import Vector

print("=== INSPECT")
for o in bpy.data.objects:
    line = f"{o.name!r} type={o.type} parent={o.parent.name if o.parent else None} loc={tuple(round(v, 3) for v in o.location)} scale={tuple(round(v, 3) for v in o.scale)}"
    if o.type == "MESH":
        me = o.data
        ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
        mn = [min(v[i] for v in ws) for i in range(3)]
        mx = [max(v[i] for v in ws) for i in range(3)]
        line += f" verts={len(me.vertices)} faces={len(me.polygons)} bbox_min={[round(x, 3) for x in mn]} bbox_max={[round(x, 3) for x in mx]}"
        line += f" materials={[m.name if m else None for m in me.materials]} modifiers={[m.type for m in o.modifiers]}"
    print(line)
for m in bpy.data.materials:
    tex = [n.image.name for n in (m.node_tree.nodes if m.use_nodes and m.node_tree else []) if n.type == "TEX_IMAGE" and n.image]
    print(f"material {m.name!r} textures={tex}")
for im in bpy.data.images:
    print(f"image {im.name!r} file={im.filepath!r} size={tuple(im.size)} packed={bool(im.packed_file)}")

# preview render (front, workbench)
scene = bpy.context.scene
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector([min(p[i] for p in pts) for i in range(3)]); mx = Vector([max(p[i] for p in pts) for i in range(3)])
ctr = (mn + mx) / 2; size = max(mx - mn)
cam_data = bpy.data.cameras.new("PreviewCam"); cam = bpy.data.objects.new("PreviewCam", cam_data)
scene.collection.objects.link(cam)
cam.location = ctr + Vector((0, -2.2 * size, 0.15 * size))
cam.rotation_euler = (1.5708, 0, 0)
scene.camera = cam
engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in engines else scene.render.engine
try:
    scene.display.shading.color_type = "TEXTURE"
except Exception:
    pass
scene.render.resolution_x = 600; scene.render.resolution_y = 800
scene.render.filepath = "C:/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/tree_preview.png"
bpy.ops.render.render(write_still=True)
print("bbox all", [round(x, 3) for x in mn], [round(x, 3) for x in mx])
print("=== DONE")
