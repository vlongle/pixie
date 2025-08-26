from __future__ import annotations
import argparse, json, os, sys, math
from pathlib import Path
import bpy
from mathutils import Matrix, Vector
import subprocess
import socket
from statistics import mean



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





def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)




# ───────────────────────────── CLI ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--obj_id",   type=str, required=True)
    p.add_argument("--frame_id", type=int, required=False, default=None,
                   help="If omitted, all frames in the ply_files folder will be rendered and combined into a video.")
    p.add_argument("--ply_dir", type=str, required=True,
                   help="Directory containing the ply files")

    # Original flag (name & default kept intact)
    p.add_argument("--blend_file_path", "--blend", dest="blend",
                   required=True,
                   help="Path to the base .blend scene")

    p.add_argument("--transparent", action="store_true")
    p.add_argument("--camera_id", type=int, default=None,
                   help="Index in transforms.json (0-based)")
    # ───────────── Object positioning & orientation ──────────────
    # Initial translation (XYZ). After import, the object is first rotated, then
    # shifted in the X and Y directions to this target location, and finally
    # dropped so that its lowest point sits exactly at the requested Z value.
    p.add_argument("--init_xyz", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="Initial [x y z] translation of the object before grounding (default: 0 0 0)")

    # User-specified Euler rotation (XYZ, in DEGREES) applied to the object
    # right after import.  Internally converted to radians.
    p.add_argument("--xyz_rotation", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                   metavar=("RX", "RY", "RZ"),
                   help="Euler XYZ rotation IN DEGREES to apply to the object (default: 0 0 0)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for rendered images")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Data directory containing transforms.json")
    p.add_argument("--place_on_ground", action="store_true",
                   help="Whether to automatically place object on ground")
    p.add_argument("--blender_gs_addon_path", type=str, required=True,
                   help="Path to the Gaussian Splatting Blender addon zip file")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing frames")

    # Optional solid background colour (0-1 floats).  Provide 3 numbers (RGB) or
    # 4 numbers (RGBA).  If omitted, the background from the .blend file is
    # left untouched.
    p.add_argument("--bg_color", nargs="*", type=float, default=None,
                   help="Optional RGB(A) background colour (0-1 floats). "
                        "Give 3 or 4 values; omit flag to keep the scene default.")

    # Camera orbit: degrees of rotation (around world Z) applied *per frame* 
    # relative to the camera orientation defined by transforms.json (or the
    # default scene camera).  The camera orbits around an empty named
    # "dolly_focus" that must exist in the .blend file.
    p.add_argument("--rotate_around", type=float, default=None,
                   help="Degrees of orbit around 'dolly_focus' per frame. "
                        "E.g. 10 rotates the camera 10 degrees each frame, for a 36-frame full 360°.")
    p.add_argument("--frame_offset", type=int, default=0,
                   help="Offset the frame index by this amount. "
                        "E.g. 1 shifts the frame index by 1, so the render will use frame 1 instead of 0.")
    
    # ───────────── Render quality settings ─────────────
    p.add_argument("--resolution_x", type=int, default=None,
                   help="Render resolution X (width). If omitted, uses scene default.")
    p.add_argument("--resolution_y", type=int, default=None,
                   help="Render resolution Y (height). If omitted, uses scene default.")
    p.add_argument("--cycles_samples", type=int, default=None,
                   help="Number of Cycles render samples. If omitted, uses scene default.")

    p.add_argument("--num_renders", type=int, default=None,
                   help="Number of renders to do. If omitted, uses scene default.")
    p.add_argument("--start_frame", type=int, default=0,
                   help="Start frame to render. If omitted, uses scene default.")
    p.add_argument("--is_dropping", action="store_true",
                   help="Whether to drop the object on the ground. If omitted, uses scene default.")
    # Debugging: optionally save a .blend snapshot per frame
    p.add_argument("--save_blend", action="store_true",
                   help="Save a .blend file (same name as output PNG) for each rendered frame.")
    p.add_argument("--focal_length_ratio", type=float, default=None,
                   help="Ratio of camera focal length to original focal length. "
                        "E.g. 1.2 scales the focal length by 1.2x.")
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:])

# ──────────────────────── Path helpers ─────────────────────────
def transforms_json(obj_id: str, data_dir: str) -> Path:
    return Path(data_dir) / "transforms.json"

def output_png(obj_id: str, frame_id: int, output_folder: str) -> Path:
    return Path(output_folder) / f"frame_{frame_id:05d}.png"

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

# ───────────────────── Misc. small utilities ───────────────────
def shift_xyz(obj, xyz_target=(0.0, 0.0, 0.0)):
    obj.location = Vector((xyz_target[0], xyz_target[1], xyz_target[2]))
    return obj

def place_on_ground(obj, xy_target=(0.0, 0.0), ground_z=0.0, clearance=0.0):
    """Drop object so its lowest point is at ground_z (plus clearance)."""
    bpy.context.view_layer.update()
    zmin = min((obj.matrix_world @ Vector(c)).z for c in obj.bound_box)
    # obj.location = Vector((xy_target[0], xy_target[1],
    #                        obj.location.z + (ground_z + clearance - zmin)))
    zraise = ground_z + clearance - zmin
    xyz = (xy_target[0], xy_target[1], zraise)
    obj = shift_xyz(obj, xyz)
    return zraise 

# ─────────────────── Batch & video utilities ──────────────────
def compile_video(img_dir: Path, fps: int = 10, out_name: str = "render.mp4"):
    """Stitch PNG frames in `img_dir` into an MP4 using ffmpeg."""
    video_path = img_dir / out_name
    pattern    = img_dir / "frame_%05d.png"
    cmd = [
        "ffmpeg", "-y",               # overwrite without prompt
        # "/bin/ffmpeg", "-y",               # overwrite without prompt
        "-framerate", str(fps),       # input FPS
        "-i", str(pattern),           # input pattern
        "-c:v", "libx264",           # video codec
        "-pix_fmt", "yuv420p",       # widely-compatible pixel format
        str(video_path)
    ]
    print("🎞️  Creating video:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Video saved to {video_path}")
    except FileNotFoundError:
        print("⚠️  ffmpeg not found; skipping video compilation.")


def _ensure_gaussian_splat_addon(zip_path: str):
    """Ensure the Gaussian-Splatting Blender add-on is enabled.

    Tries to enable an already installed add-on first to avoid race conditions
    when many batch jobs start concurrently.  If enabling fails, attempts to
    (re)install from *zip_path* and enable again.  Should be safe to call
    repeatedly.
    """
    import addon_utils

    try:
        # Try to enable if installed but not enabled
        bpy.ops.preferences.addon_enable(module="blender-addon")
        return                                    # success
    except Exception as enable_exc:
        print(f"⚠️  Could not enable existing 'blender-addon': {enable_exc}\n→ Attempting reinstall…")

    # Remove any broken install to start clean
    try:
        bpy.ops.preferences.addon_remove(module="blender-addon")
    except Exception:
        pass  # OK if it wasn't there

    # (Re)install from ZIP and enable
    try:
        bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
        bpy.ops.preferences.addon_enable(module="blender-addon")
        print("✅ 'blender-addon' installed & enabled successfully.")
    except Exception as install_exc:
        print(f"❌ Failed to install/enable 'blender-addon': {install_exc}")
        raise

def _orbit_camera(cam: bpy.types.Object, angle_deg_per_frame: float, frame_id: int):
    """Orbit *cam* around Z-axis by *angle_deg_per_frame* × *frame_id*.

    If an object named 'dolly_focus' exists, orbit around its location using
    explicit vector math (keeps camera looking at the focus).  Otherwise, fall
    back to the pivot-parent method: create/reuse an empty named
    '_orbit_pivot' at the world origin, parent the camera to it (keeping
    transforms), and rotate the pivot.
    """
    angle_rad = math.radians(angle_deg_per_frame * frame_id)

    focus = bpy.data.objects.get("dolly_focus")
    if focus is not None:
        # ---------------- Focus-based orbit (existing behaviour) -------------
        cam_vec = cam.matrix_world.translation - focus.location
        new_vec = Matrix.Rotation(angle_rad, 4, 'Z') @ cam_vec
        cam.matrix_world.translation = focus.location + new_vec

        # Make camera look at the focus point
        dir_vec = (focus.location - cam.matrix_world.translation).normalized()
        cam.rotation_mode = 'XYZ'
        cam.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
        print(f"⭐ Orbit via 'dolly_focus'   angle={angle_deg_per_frame*frame_id:.2f}°  cam={cam.location}")
    else:
        # ---------------- Pivot-parent orbit (render_blender_qual_paper style)
        print("⭐ Initial camera location", cam.location)
        pivot = bpy.data.objects.get("_orbit_pivot")
        if pivot is None:
            # cam_height = cam.location.z
            # bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, cam_height))
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
            pivot = bpy.context.active_object
            pivot.name = "_orbit_pivot"
            print("⭐ Created _orbit_pivot empty at origin for camera orbit")

        # Parent camera once (keeping its current world transform)
        # world_mtx = cam.matrix_world.copy()
        # cam.parent = pivot
        # cam.matrix_parent_inverse = pivot.matrix_world.inverted()
        # cam.matrix_world = world_mtx

        print("created pivot at location", pivot.location)
        cam.parent = pivot

        pivot.rotation_mode = 'XYZ'
        pivot.rotation_euler[2] = angle_rad  # rotate around Z-axis
        print(f"⭐ Orbit via pivot         angle={angle_deg_per_frame*frame_id:.2f}°  cam={cam.location}")


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


def _debug_print_obj_bounds(obj):
    """Print world-space bounding box and centre of *obj* for debugging."""
    import mathutils
    world_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in world_corners])
    centre = mathutils.Vector((mean(xs), mean(ys), mean(zs)))
    mins   = (min(xs), min(ys), min(zs))
    maxs   = (max(xs), max(ys), max(zs))
    print(f"🔍 OBJ BBOX   min={mins}  max={maxs}")
    print(f"🔍 OBJ CENTRE {centre}")

def render_single_frame(args: argparse.Namespace, *, load_blend: bool = True):
    """Render a single frame based on the already-parsed `args`.

    Parameters
    ----------
    load_blend : bool, optional
        If True, reload the base .blend file before rendering.  For batch
        rendering of many frames this can be set to False after the first
        frame to avoid the expensive scene reload, drastically speeding up
        subsequent renders.
    """


    # BG_COLOR = parse_color("#f5f5f5")
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



    # Resolve all paths
    ply_dir = Path(args.ply_dir)
    assert ply_dir.exists(), f"PLY directory does not exist: {ply_dir}"
    ply = ply_dir / f"frame_{args.frame_id:05d}.ply"
    assert ply.exists(), f"PLY file does not exist: {ply}"
    tf_json  = transforms_json(args.obj_id, args.data_dir)
    out_png  = output_png(args.obj_id, args.frame_id, args.output_dir)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Skip if frame already rendered
    if out_png.exists() and not args.overwrite:
        print(f"⏭️  Skipping frame {args.frame_id}: {out_png.name} already exists.")
        return

    # Load clean .blend (only if requested) -----------------------------------
    if load_blend:
        bpy.ops.wm.open_mainfile(filepath=args.blend)
        # Ensure any key-framed motion is evaluated for this frame.
        bpy.context.scene.frame_set(args.frame_id + args.frame_offset)

        # Ensure the add-on providing the import operator is active
        _ensure_gaussian_splat_addon(args.blender_gs_addon_path)
    else:
        # Scene was opened previously; just jump to the correct frame.
        bpy.context.scene.frame_set(args.frame_id + args.frame_offset)

    # ------------------------------------------------------------------------
    # Import Gaussian-Splat for this frame
    bpy.ops.object.import_gaussian_splatting(filepath=str(ply.resolve()))
    gs_obj = bpy.context.selected_objects[-1]

    gs_obj.rotation_mode  = 'XYZ'              # make sure we're in XYZ Euler
    gs_obj.rotation_euler = tuple(math.radians(a) for a in args.xyz_rotation)

    print("args.init_xyz", args.init_xyz)
    print("args.xyz_rotation", args.xyz_rotation)
    print("args.place_on_ground", args.place_on_ground)
    
    if args.place_on_ground:
        # Drop the object so its lowest point is exactly at init_xyz[2]
        zraise = place_on_ground(
            gs_obj,
            xy_target=(args.init_xyz[0], args.init_xyz[1]),
            ground_z=args.init_xyz[2],
            clearance=0.0,
        )
        print(f"frame {args.frame_id} zraise: {zraise}")
    else:
        # Manual positioning - use init_xyz directly
        zraise = args.init_xyz[2] if any(args.init_xyz) else 1.5
        shift_xyz(gs_obj, (args.init_xyz[0], args.init_xyz[1], zraise))

    print(f"frame {args.frame_id} zraise: {zraise}")

    # Full-splats rendering (disable point-cloud preview)
    gn_nodes  = gs_obj.modifiers["Geometry Nodes"].node_group.nodes
    bool_node = gn_nodes.get("Boolean")
    if bool_node:
        bool_node.boolean = False  # full-splat mode

    # Ensure 100 % of the splats are displayed (quality over speed)
    rv_node = gn_nodes.get("Random Value")
    def_val = rv_node.inputs["Probability"].default_value
    if rv_node and rv_node.inputs.get("Probability") is not None:
        rv_node.inputs["Probability"].default_value = 1.0


    _debug_print_obj_bounds(gs_obj)
    # Camera
    cam = bpy.context.scene.camera
    if args.camera_id is not None:
        apply_camera(cam, tf_json, args.camera_id)
        cam.matrix_world.translation.z += zraise

    
    if args.focal_length_ratio is not None:
        cam.data.lens_unit = 'MILLIMETERS'
        print(f"current focal length: {cam.data.lens}mm. Setting by ratio {args.focal_length_ratio}")
        cam.data.lens = args.focal_length_ratio * cam.data.lens
        print(f"Set camera focal length to {args.focal_length_ratio * cam.data.lens}mm")


    # ───────────── Optional camera orbit around "dolly_focus" ─────────────
    if args.rotate_around is not None:
        _orbit_camera(cam, args.rotate_around, args.frame_id)

    # Render settings
    scn = bpy.context.scene
    ## use GPU for rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.cycles.device = 'CPU'
    
    # Apply custom render settings if provided
    if args.cycles_samples is not None:
        scn.cycles.samples = args.cycles_samples
    else:
        scn.cycles.samples = 16
    # scn.cycles.samples = 160
    # scn.cycles.samples = 4096
    
    if args.resolution_x is not None:
        scn.render.resolution_x = args.resolution_x
    else:
        scn.render.resolution_x = 1280
    if args.resolution_y is not None:
        scn.render.resolution_y = args.resolution_y
    else:
        scn.render.resolution_y = 720

    scn.render.film_transparent = args.transparent
    scn.render.resolution_x = scn.render.resolution_y = 512
    scn.view_settings.view_transform = 'Filmic'
    scn.view_settings.look = 'Very High Contrast'
    # scn.view_settings.view_transform = 'Standard'
    # scn.view_settings.look = 'None'
    scn.render.filepath = str(out_png)

    # ───────────── Optional background colour via Spot light ─────────────
    if args.bg_color:
        bg_col = tuple(args.bg_color)
        if len(bg_col) == 4:
            rgb = bg_col[:3]
        elif len(bg_col) == 3:
            rgb = bg_col
        else:
            raise ValueError("--bg_color expects 3 or 4 floats (RGB[A]).")

        # world           = bpy.context.scene.world
        # world.use_nodes = True
        # ntree           = world.node_tree
        # ntree.nodes.clear()

        # # Camera-visible background
        # bg_visible = ntree.nodes.new("ShaderNodeBackground")
        # bg_visible.location = (-200,  50)
        # bg_visible.inputs["Color"].default_value    = bg_col
        # bg_visible.inputs["Strength"].default_value = 1.0

        # # Lighting-only background (white)
        # bg_light = ntree.nodes.new("ShaderNodeBackground")
        # bg_light.location = (-200, -150)
        # bg_light.inputs["Color"].default_value    = (1.0, 1.0, 1.0, 1.0)
        # bg_light.inputs["Strength"].default_value = 1.0

        # path = ntree.nodes.new("ShaderNodeLightPath")
        # path.location = (-400, -50)

        # mix  = ntree.nodes.new("ShaderNodeMixShader")
        # mix.location = (50, -50)

        # out = ntree.nodes.new("ShaderNodeOutputWorld")
        # out.location = (250, -50)

        # # Link nodes to replicate animate_tree.py set-up
        # ntree.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        # ntree.links.new(bg_visible.outputs["Background"], mix.inputs[2])
        # ntree.links.new(bg_light.outputs["Background"],  mix.inputs[1])
        # ntree.links.new(mix.outputs["Shader"],           out.inputs["Surface"])


        spot = bpy.data.objects.get("Spot")
        if spot and hasattr(spot.data, "color"):
            spot.data.color = rgb
        else:
            print("⚠️  --bg_color given but no light named 'Spot' found with color attribute; skipping.")

        ## crank the power up by 10x
        spot.data.energy = 2500

    # Go!
    bpy.ops.render.render(write_still=True)
    print(f"✅ Saved PNG to {scn.render.filepath}")

    # Optionally write a .blend snapshot for debugging before cleaning up.
    if args.save_blend:
        blend_path = out_png.with_suffix(".blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"💾 Saved .blend snapshot to {blend_path}")

    # Clean up the imported GS object to free memory before next frame.
    bpy.data.objects.remove(gs_obj, do_unlink=True)

# ───────────────────────── Main routine ───────────────────────
def main():
    args = parse_args()

    # If a specific frame is requested, render just that one.
    if args.frame_id is not None:
        render_single_frame(args)
        return

    # ───── Batch mode: find all frame_*.ply files and render them ─────
    ply_dir = Path(args.ply_dir)
    assert ply_dir.exists(), f"Could not locate ply_files directory at {ply_dir}"

    frame_ids = sorted(int(p.stem.split("_")[1]) for p in ply_dir.glob("frame_*.ply"))
    if not frame_ids:
        print(f"Error: No frame_*.ply files found in {ply_dir}. Aborting. Did you run `pipeline.py` 
        with `physics.save_ply=true`? 
        Alternatively, comment out `blender_gs` in `output_rendering/default.yaml`")
        sys.exit(1)
    ## HACK: to ensure rotation is consistent over different videos of different frames
    ## assume that args.rotate_around is the TOTAL over all frames
    if args.rotate_around is not None:
        args.rotate_around = args.rotate_around / len(frame_ids)
        print(f"args.rotate_around: {args.rotate_around}")
    if args.num_renders is not None:
        frame_ids = frame_ids[args.start_frame:args.start_frame+args.num_renders]

    print(f"📄 Found {len(frame_ids)} frames to render: {frame_ids[0]} … {frame_ids[-1]}")

    for idx, fid in enumerate(frame_ids):
        args.frame_id = fid
        print(f"[...] Rendering frame {fid} / {frame_ids[-1]}")
        render_single_frame(args, load_blend=True)

# -------------------------------------------------------------------------
# __main__ guard
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()


