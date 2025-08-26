import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import bpy
from mathutils import Vector
import shutil
import json
import objaverse
import socket



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

enable_cuda_devices()





def get_default_output_dir(format_type):
    """Determine the default output directory based on the format."""
    home_dir = os.path.expanduser("~")
    if format_type == "NGP":
        return os.path.join(home_dir, "code", "instant-ngp", "data")
    else:  # NERF format for gaussian splatting
        return os.path.join(home_dir, "code", "gaussian-splatting", "data")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_id",
    type=str,
    help="Objaverse object ID to process",
)
parser.add_argument(
    "--obj_path",
    type=str,
    help="Path to the object file (alternative to obj_id)",
)
parser.add_argument("--output_dir", type=str, default=None, 
                   help="Path to output directory. If not provided, will use format-specific default location.")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist_min", type=float, default=1.0, help="Minimum camera distance")
parser.add_argument("--camera_dist_max", type=float, default=1.4, help="Maximum camera distance")
# Keep camera_dist for backward compatibility
parser.add_argument("--camera_dist", type=float, default=1.2, help="Camera distance (deprecated, use min/max instead)")
parser.add_argument("--format", type=str, default="NERF", choices=["NERF", "NGP"])
parser.add_argument("--transparent_bg", action='store_true', help="Render with transparent background")
parser.add_argument("--scene_scale", type=float, default=1.0, help="Scale factor to apply after normalization")
parser.add_argument("--blender_nerf_addon_path", type=str, required=True, help="Path to the BlenderNeRF addon zip file")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

if args.obj_id is None and args.obj_path is None:
    raise ValueError("Either --obj_id or --obj_path must be provided")

# If obj_id is provided, get the object path from Objaverse
if args.obj_id is not None:
    print(f"Looking up object path for ID: {args.obj_id}")
    objects = objaverse.load_objects(uids=[args.obj_id])
    if not objects or args.obj_id not in objects:
        raise ValueError(f"Could not find object with ID: {args.obj_id}")
    args.obj_path = objects[args.obj_id]
    print(f"Found object path: {args.obj_path}")



# Set the output directory if not provided
if args.output_dir is None:
    args.output_dir = get_default_output_dir(args.format)
    print(f"Using default output directory: {args.output_dir}")

# Install the BlenderNeRF addon
print(f"Installing BlenderNeRF addon from: {args.blender_nerf_addon_path}")
bpy.ops.preferences.addon_install(filepath=args.blender_nerf_addon_path, overwrite=True)
bpy.ops.preferences.addon_enable(module='BlenderNeRF-main-custom')

context = bpy.context
scene = context.scene
render = scene.render


## configure rendering settings
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
# render.resolution_x = 800
# render.resolution_y = 800
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True


### extra settings to ensure pure white background for gaussian splatting...
# scene.world.light_settings.use_ambient_occlusion = False
scene.world.use_nodes = True
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'

scene.render.film_transparent = args.transparent_bg

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )



def add_lighting() -> None:
    """Add a professional studio-like lighting setup with multiple area lights."""
    # Delete the default light
    if "Light" in bpy.data.objects:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    
    # Clear any existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create a three-point lighting setup
    
    # 1. Key light (main light) - brightest, from front-right
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 2))
    key_light = bpy.context.object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 5
    key_light.rotation_euler = (0.6, 0.2, 0.8)  # Angle toward the subject
    
    # 2. Fill light - softer light from opposite side to fill shadows
    bpy.ops.object.light_add(type="AREA", location=(-2, -1, 1))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200  # Less intense than key light
    fill_light.data.size = 7  # Larger for softer light
    fill_light.rotation_euler = (0.5, -0.2, -0.8)
    
    # 3. Rim/Back light - creates separation from background
    bpy.ops.object.light_add(type="AREA", location=(0, 3, 2))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 300
    rim_light.data.size = 4
    rim_light.rotation_euler = (0.8, 0, 0)  # Point down at the back of subject
    
    # 4. Top light for general fill
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4))
    top_light = bpy.context.object
    top_light.name = "Top_Light"
    top_light.data.energy = 150
    top_light.data.size = 10
    top_light.rotation_euler = (0, 0, 0)  # Point straight down
    
    if not args.transparent_bg:
        world = bpy.data.worlds['World']
        world.use_nodes = True
        bg_node = world.node_tree.nodes['Background']
        bg_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0) ## gray
        bg_node.inputs[1].default_value = 1.0  # Full strength



def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)





# load the model
def load_object(obj_path: str) -> None:
    """Loads a 3D model into the scene."""
    if obj_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=obj_path, merge_vertices=True)
    elif obj_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=obj_path)
    elif obj_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=obj_path)
    else:
        raise ValueError(f"Unsupported file type: {obj_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * args.scene_scale
    
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    
    bpy.ops.object.select_all(action="DESELECT")

def setup_manual_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    return cam, cam_constraint

def render_with_blendernerf(object_uid: str) -> None:
    """Use BlenderNerf add-on to render the normalized scene."""
    # args.output_dir already contains the object_uid from the pipeline
    output_dir = args.output_dir
    
    # Clear the output directory if it exists to remove stale data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up parameters for the BlenderNeRF add-on
    scene = bpy.context.scene
    
    # Global parameters
    scene.train_data = True
    scene.test_data = False
    # scene.aabb = 4  # Smaller bounding box to focus on the object
    # scene.aabb = 2 # Smaller bounding box to focus on the object
    scene.aabb = 32
    scene.render_frames = True
    scene.nerf = args.format == "NERF"  # True for NeRF format, False for NGP format
    scene.save_path = output_dir
    
    # COS specific parameters
    scene.cos_dataset_name = object_uid  # Use object_uid as dataset name to avoid nested directories
    scene.sphere_location = (0.0, 0.0, 0.0)  # Centered at origin after normalization
    scene.sphere_rotation = (0.0, 0.0, 0.0)
    scene.sphere_scale = (1.0, 1.0, 1.0)
    
    # Use the new radius min/max properties if available in the add-on
    # Otherwise fall back to the original sphere_radius property
    if hasattr(scene, 'sphere_radius_min') and hasattr(scene, 'sphere_radius_max'):
        scene.sphere_radius_min = args.camera_dist_min / 2
        scene.sphere_radius_max = args.camera_dist_max / 2
        # Set sphere_radius to the average for visualization
        scene.sphere_radius = (args.camera_dist_min + args.camera_dist_max) / 4
    else:
        # Fallback to original behavior
        scene.sphere_radius = args.camera_dist / 2
        
    scene.focal = 20.0  # lens focal length in mm
    scene.cos_nb_frames = args.num_images
    scene.seed = 0
    scene.upper_views = True
    scene.outwards = False
    
    try:
        # Run the Camera on Sphere operator from BlenderNerf
        bpy.ops.object.camera_on_sphere()
        print(f"Successfully rendered {args.num_images} images using BlenderNerf add-on")
    except Exception as e:
        # Check if this is the harmless "BlenderNeRF Camera" not found error
        if "BlenderNeRF Camera" in str(e) and "not found" in str(e):
            print(f"Warning: Harmless BlenderNeRF camera error (continuing): {e}")
            # The rendering actually completed successfully despite this error
        else:
            print(f"Error during BlenderNerf rendering: {e}")
            raise


    # Unpack the archive - BlenderNeRF creates a zip with the dataset name
    zip_path = os.path.join(output_dir, f"{object_uid}.zip")
    if os.path.exists(zip_path):
        shutil.unpack_archive(zip_path, output_dir)
        os.remove(zip_path)

    if args.format == "NERF":
        # create a dummy transforms_test.json
        with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
            json.dump({"camera_angle_x": 0.0, "frames": []}, f)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path

def process_object(obj_path: str) -> None:
    """Process a single object: load, normalize, and render."""
    reset_scene()
    
    # Load the object
    load_object(obj_path)
    object_uid = os.path.basename(obj_path).split(".")[0]
    
    # Normalize the scene
    normalize_scene()
    print(f"Scene normalized for {object_uid}")

    # Add lighting
    add_lighting()
    
    # Render with BlenderNerf add-on (this will create the directory structure)
    render_with_blendernerf(object_uid)
    
    # Save the normalized scene as GLB in the output directory
    # args.output_dir already contains the object_uid from the pipeline
    glb_path = os.path.join(args.output_dir, f"{object_uid}_normalized_scene.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",          # binary container
        export_apply=False,           # keep object transforms
        export_texcoords=True,
        export_normals=True,
        export_attributes=True,       # ← replaces export_colors
        export_materials='EXPORT',    # keep Principled BSDF
        export_animations=False,
        check_existing=False,
    )
    print(f"Normalized scene exported to {glb_path}")

if __name__ == "__main__":
    try:
        start_i = time.time()
        
        if args.obj_path.startswith("http"):
            local_path = download_object(args.obj_path)
        else:
            local_path = args.obj_path
        
        process_object(local_path)
        
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        
        # Delete the object if it was downloaded
        if args.obj_path.startswith("http"):
            os.remove(local_path)
            
    except Exception as e:
        print("Failed to process", args.obj_path)
        print(e)

