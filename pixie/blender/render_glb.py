import bpy
import os
import sys
import argparse
from mathutils import Vector, Matrix
from pathlib import Path
import json
import socket
import contextlib  # needed by the helper below
import math
import shutil




def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated

def _srgb_to_linear(c: float) -> float:
    """Convert a single channel from sRGB (display) to linear space (Blender)."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def parse_color(s: str | tuple):
    """Return an (R,G,B,A) tuple in *linear* space ready for Blender.

    Accepts either an existing 4-tuple (assumed already linear) or a hex string
    like "#f5f5f5" or "#fff" (sRGB).  Hex input is converted to linear using
    the standard sRGB→linear transform so that the *saved PNG* matches the
    intended display colour exactly.
    """
    if isinstance(s, tuple):
        # Already linear RGBA
        return s if len(s) == 4 else (*s, 1.0)

    s = s.lstrip('#')
    if len(s) == 3:                         # short form e.g. fff
        s = ''.join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError("Colour must be #rgb or #rrggbb")

    # Convert 0–255 integers → 0–1 floats (sRGB)
    sr, sg, sb = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # sRGB → linear so Blender writes the same sRGB after colour-management
    lr, lg, lb = map(_srgb_to_linear, (sr, sg, sb))
    return (lr, lg, lb, 1.0)


enable_cuda_devices()

parser = argparse.ArgumentParser()
# parser.add_argument("--frame_blend_path", type=str)
parser.add_argument("--frame", type=int)
parser.add_argument("--blend_file_path", type=str, 
                    required=True,)
parser.add_argument('--obj', required=True, type=str)
parser.add_argument('--obj_id', required=True, type=str)
parser.add_argument('--output_folder', required=True, type=str)
parser.add_argument('--views', type=int)
parser.add_argument('--resolution', type=int)
parser.add_argument('--input_model', type=str)
parser.add_argument('--scene_scale', type=float, default=1.0)
parser.add_argument('--transparent', action='store_true')
parser.add_argument('--rotate_video', action='store_true',
                    help='If set, instead of a single still, render a full 360° rotation around the object.')
parser.add_argument('--camera_id', type=int, default=None, help='Camera index in transforms.json (0-based, optional)')
parser.add_argument('--focal_length', type=float, default=None,
                    help='Set camera focal length in mm. Overrides intrinsics from transforms.json if provided.')
parser.add_argument('--data_dir', type=str, default=None,
                    help='Data directory containing the transforms.json file')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

def transforms_json(obj_id: str, data_dir: str = None) -> Path:
    return Path(data_dir) / "transforms.json"


# ------------------------------------------------------------------
# Patch translucent materials so they *do* cast shadows in Cycles
# ------------------------------------------------------------------

def _ensure_shadow_cast(mat: bpy.types.Material):
    """If the material uses alpha-blend/hashed, add a Light-Path mix
    that forces the surface to be opaque for shadow rays while leaving
    its usual appearance for camera rays."""

    if not mat.use_nodes:
        return

    if mat.blend_method not in {'BLEND', 'HASHED', 'CLIP'}:
        return  # already opaque → casts shadows

    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if not out or not out.inputs['Surface'].is_linked:
        return

    # Avoid patching twice
    if nt.nodes.get('_AutoLP') and any(n.type == 'MIX_SHADER' and n.label == '_ShadowFix' for n in nt.nodes):
        return

    # Create Light Path node (or reuse if extant)
    lp = nt.nodes.get('_AutoLP') or nt.nodes.new('ShaderNodeLightPath')
    lp.name = '_AutoLP'
    lp.location = (-300, -300)

    # Original surface shader
    orig_link = out.inputs['Surface'].links[0]
    orig_socket = orig_link.from_socket

    # Build or retrieve an opaque BSDF.
    # Prefer an existing Principled node from the tree (most GLBs have one).
    opaque_shader = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)

    if opaque_shader is None:
        opaque_shader = nt.nodes.new('ShaderNodeBsdfPrincipled')
        opaque_shader.location = lp.location.x + 100, lp.location.y - 200

    # Force full opacity
    if hasattr(opaque_shader.inputs, 'Alpha'):
        opaque_shader.inputs['Alpha'].default_value = 1.0

    # Mix Shader that swaps based on shadow ray
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.label = '_ShadowFix'
    mix.location = (orig_link.from_node.location.x + 200, orig_link.from_node.location.y)

    nt.links.new(orig_socket, mix.inputs[1])   # regular shading
    nt.links.new(opaque_shader.outputs[0], mix.inputs[2])  # opaque branch
    nt.links.new(lp.outputs['Is Shadow Ray'], mix.inputs['Fac'])

    # Re-wire to material output
    nt.links.new(mix.outputs[0], out.inputs['Surface'])




# ─────────────────── Camera configuration ─────────────────────
def set_intrinsics(cam: bpy.types.Object, intr: dict):
    camd = cam.data
    camd.lens_unit = 'FOV'
    camd.angle_x   = intr["camera_angle_x"]
    camd.angle_y   = intr["camera_angle_y"]

    # cx,cy shift → Blender shift_x/y  (±0.5 == half the image width/height)
    w, h = intr["w"], intr["h"]
    camd.shift_x =  (intr["cx"] - 0.5 * w) / w
    camd.shift_y = -(intr["cy"] - 0.5 * h) / h  # y-axis is flipped

def set_extrinsics(cam: bpy.types.Object, mat4: list[list[float]]):
    cam.matrix_world = Matrix(mat4)

def apply_camera(cam: bpy.types.Object, tf_path: Path, cam_id: int):
    data = json.loads(tf_path.read_text())
    frames = data["frames"]
    if not (0 <= cam_id < len(frames)):
        raise IndexError(f"camera_id {cam_id} out of range 0..{len(frames)-1}")
    set_extrinsics(cam, frames[cam_id]["transform_matrix"])
    set_intrinsics(cam, data)



def scene_bbox(mesh_objects):
    """Calculate bounding box for all mesh objects."""
    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in mesh_objects:
        for corner in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner)
            bbox_min.x = min(bbox_min.x, corner_world.x)
            bbox_min.y = min(bbox_min.y, corner_world.y)
            bbox_min.z = min(bbox_min.z, corner_world.z)
            bbox_max.x = max(bbox_max.x, corner_world.x)
            bbox_max.y = max(bbox_max.y, corner_world.y)
            bbox_max.z = max(bbox_max.z, corner_world.z)
    
    return bbox_min, bbox_max


def normalize_scene(mesh_objs, scene_scale=1.0, raise_to_ground=True,
zoffset=0.05):
    bbox_min, bbox_max = scene_bbox(mesh_objs)
    scale = 1 / max(bbox_max - bbox_min)
    for obj in mesh_objs:
        obj.scale = obj.scale * scale * scene_scale
    
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    
    # Get updated bounding box after scaling
    bbox_min, bbox_max = scene_bbox(mesh_objs)
    
    if raise_to_ground:
        # Move object so its bottom is at z=0
        offset = Vector((0, 0, -bbox_min.z+zoffset))
    else:
        # Center the scene at the origin
        offset = -(bbox_min + bbox_max) / 2

    print("Raise to ground?", raise_to_ground, "offset", offset)
    for obj in mesh_objs:
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    
    # Verify final bounding box
    final_min, final_max = scene_bbox(mesh_objs)
    final_size = final_max - final_min
    print(f"  Final bounding box min: {final_min}")
    print(f"  Final bounding box max: {final_max}")
    print(f"  Final bounding box size: {final_size}")
    return mesh_objs




bpy.ops.wm.open_mainfile(filepath=args.blend_file_path)
scene = bpy.context.scene
pre_import_meshes = {obj for obj in scene.objects if obj.type == 'MESH'}
bpy.ops.import_scene.gltf(filepath=args.obj)

imported_meshes = [
    obj for obj in scene.objects
    if obj.type == 'MESH' and obj not in pre_import_meshes
]



# ---------------------------- DEBUG LOGGING -----------------------------
normalize_scene(imported_meshes, scene_scale=args.scene_scale, raise_to_ground=True)


# Apply the fix to imported materials --------------------------------
for obj in imported_meshes:
    for mat in obj.data.materials:
        if mat:
            _ensure_shadow_cast(mat)

# ------------------------------------------------------------------
# 3 – floor: ultra-glossy plane
# ------------------------------------------------------------------
plane = bpy.data.objects.get("Plane")
if plane is None:
    raise RuntimeError("No object named 'Plane' found!")

plane.cycles.is_shadow_catcher = True          # Cycles only
# plane.cycles.is_shadow_catcher = False
## delete the plane
if args.transparent:
    bpy.data.objects.remove(plane, do_unlink=True)




# BG_COLOR = (0.878, 0.498, 0.867, 1.0)
# BG_COLOR = (1.0, 1.0, 1.0, 1.0) ## color that the camera actually seees
# rgba(245,245,245,255)
# BG_COLOR = parse_color("#f5f5f5")
# # BG_COLOR = parse_color("#154c79")



# ## color for environment lighting
# # LIGHT_BG_COLOR = (1.0, 1.0, 1.0, 1.0)                # GI colour (keep white)
# LIGHT_BG_COLOR = parse_color("#f5f5f5")
# LIGHT_BG_STR   = 1.0                                 # world-light strength

# world           = bpy.context.scene.world
# world.use_nodes = True
# ntree           = world.node_tree
# ntree.nodes.clear()

# # nodes
# bg_visible = ntree.nodes.new("ShaderNodeBackground")
# bg_visible.location = (-200,  50)
# bg_visible.inputs["Color"].default_value    = BG_COLOR
# bg_visible.inputs["Strength"].default_value = 1.0

# bg_light = ntree.nodes.new("ShaderNodeBackground")
# bg_light.location = (-200, -150)
# bg_light.inputs["Color"].default_value    = LIGHT_BG_COLOR
# bg_light.inputs["Strength"].default_value = LIGHT_BG_STR

# path = ntree.nodes.new("ShaderNodeLightPath")
# path.location = (-400, -50)

# mix = ntree.nodes.new("ShaderNodeMixShader")
# mix.location = (50, -50)

# out = ntree.nodes.new("ShaderNodeOutputWorld")
# out.location = (250, -50)

# # links
# ntree.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
# ntree.links.new(bg_visible.outputs["Background"], mix.inputs[2])
# ntree.links.new(bg_light.outputs["Background"],  mix.inputs[1])
# ntree.links.new(mix.outputs["Shader"],           out.inputs["Surface"])




## NOTE: new for render_gs
tf_json  = transforms_json(args.obj_id, args.data_dir)
zraise = 0.5
# Camera
cam = bpy.context.scene.camera
print("Active camera", cam)
if args.camera_id is not None:
    apply_camera(cam, tf_json, args.camera_id)
    cam.matrix_world.translation.z += zraise

if args.focal_length is not None:
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = args.focal_length
    print(f"Set camera focal length to {args.focal_length}mm")




path = os.path.abspath(args.output_folder)
i_pos = 0

# Output settings
scene = bpy.context.scene
render = bpy.context.scene.render
scene.render.image_settings.file_format = 'PNG'
# scene.render.filepath = "//test_render_output"
bpy.context.scene.render.filepath = f'{path}/{str(i_pos).zfill(3)}.png'
scene.render.engine = 'CYCLES'
scene.cycles.device = "GPU"
# scene.cycles.samples = 16
scene.cycles.samples = 32
# render.resolution_x = args.resolution
# render.resolution_y = args.resolution

## 16:9 aspect ratio for ours image
# render.resolution_y = 512
# render.resolution_x = int(render.resolution_y * (16 / 9))

# scene.render.resolution_percentage = 100

## NOTE: new for render_gs for rendering the baseline
scene.render.resolution_x = scene.render.resolution_y = 512


## COLOR MANAGEMENT
scene.world.use_nodes = True
# scene.view_settings.view_transform = 'Standard'
# scene.view_settings.look = 'None'
## NOTE: new for render_gs
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Very High Contrast'
scene.render.film_transparent = args.transparent
# world.node_tree.nodes["Background"].inputs["Color"].default_value = (1,1,1,1)

def deg_to_rad(deg):
    return deg * math.pi / 180

# shutil.rmtree(path, ignore_errors=True)   # recursively delete
if args.rotate_video:
    num_frames = args.views if args.views is not None else 120

    # Create an empty at the origin to act as rotation pivot
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    pivot = bpy.context.active_object

    # Parent the camera to this pivot so rotating the pivot rotates the camera
    cam.parent = pivot


    ## mirror the logic in `get_synthetic_viz_paper_rotate.py`
    rotate_around = 60 
    rotate_around = rotate_around / num_frames

    for i in range(num_frames):
        scene.render.filepath = f'{path}/{i:03d}.png'
        # if os.path.exists(scene.render.filepath):
        #     print(f"File {scene.render.filepath} already exists, skipping")
        #     continue

        angle_deg = rotate_around * i
        print("angle_deg", angle_deg)
        angle_rad = deg_to_rad(angle_deg)
        # angle_rad = (i / num_frames) * 2 * math.pi  # full 360°
        pivot.rotation_euler[2] = angle_rad         # rotate around Z-axis
        bpy.context.view_layer.update()

        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(write_still=True)


