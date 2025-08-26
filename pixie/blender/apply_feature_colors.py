#!/usr/bin/env python

from __future__ import annotations

import argparse, os, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from plyfile import PlyData

import bpy
from mathutils import Vector, kdtree
import objaverse
import random

# Try importing matplotlib - if not available, use fallback colormaps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mathutils import Matrix
import json
import socket
import subprocess

HAS_MATPLOTLIB = True






# -----------------------------------------------------------------------------
#  PART-LABEL PALETTES
# -----------------------------------------------------------------------------

# New palettes with more distinct shades within each family.
PALETTE_ELASTICITY_NEW = [
    (0.85, 0.22, 0.22),  # Brick Red
    (1.00, 0.50, 0.20),  # Orange
    (0.98, 0.75, 0.30),  # Golden Yellow
    (0.70, 0.35, 0.10),  # Brownish-Orange
    (1.00, 0.60, 0.60),  # Light Salmon Pink
]

PALETTE_PLASTICITY_NEW = [
    (0.10, 0.40, 0.75),  # Steel Blue
    (0.20, 0.70, 0.65),  # Teal / Aquamarine
    (0.45, 0.35, 0.70),  # Indigo / Dark Lavender
    (0.05, 0.20, 0.45),  # Dark Navy
    (0.60, 0.80, 0.95),  # Sky Blue
]

# Old palettes kept for reference or if user prefers them via a future flag
_PALETTE_ELASTICITY_ORIGINAL = [
    (0.98, 0.40, 0.40),  # light red
    (0.90, 0.20, 0.20),  # red
    (0.70, 0.10, 0.10),  # dark red
    (0.50, 0.00, 0.00),  # very dark red / maroon
    (1.00, 0.60, 0.60),  # pale red
    (0.85, 0.30, 0.30),
    (0.75, 0.15, 0.15),
    (0.60, 0.05, 0.05),
]

_PALETTE_PLASTICITY_ORIGINAL = [
    (0.20, 0.60, 0.95),  # light blue
    (0.10, 0.40, 0.85),  # blue
    (0.05, 0.25, 0.65),  # dark blue
    (0.02, 0.15, 0.45),  # very dark blue
    (0.40, 0.70, 1.00),
    (0.15, 0.50, 0.90),
    (0.08, 0.30, 0.70),
    (0.04, 0.20, 0.50),
]

# Map palette-type string → colour list
PALETTE_MAP = {
    'elasticity': PALETTE_ELASTICITY_NEW,  # Using the new ones by default
    'plasticity': PALETTE_PLASTICITY_NEW,  # Using the new ones by default
    'elasticity_original':
    _PALETTE_ELASTICITY_ORIGINAL,  # For potential future choice
    'plasticity_original':
    _PALETTE_PLASTICITY_ORIGINAL,  # For potential future choice
}

# -----------------------------------------------------------------------------
#  CLI & ENV HELPERS
# -----------------------------------------------------------------------------


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool): return v
    if v.lower() in {"yes", "true", "t", "1", "y"}: return True
    if v.lower() in {"no", "false", "f", "0", "n"}: return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_argv() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Transfer CLIP-PCA colours onto mesh")
    ap.add_argument("--obj_ids",
                    nargs="+",
                    default=[],
                    type=str,
                    help="List of Objaverse UIDs of the assets")
    ap.add_argument(
        "--pred_ply",
        nargs="+",
        type=str,
        help="List of PLY paths for CLIP PCA colours",
        default=[
        ])
    ap.add_argument(
        "--clip_pred_ply",
        nargs="+",
        type=str,
        help="List of PLY paths for CLIP PCA colours",
        default=[
        ])
    ap.add_argument(
        "--glb_paths",
        nargs="+",
        type=str,
        default=None,
        help="List of local GLB paths",
    )
    ap.add_argument("--output_dir",
                    type=str,
                    default="test_qualitative_debug_viz",
                    help="Directory for coloured GLB")
    ap.add_argument("--normalize",
                    type=str2bool,
                    default=True,
                    help="Normalize mesh to unit cube")
    ap.add_argument("--scene_scale",
                    type=float,
                    default=1.0,
                    help="Scale factor after normalisation")
    ap.add_argument("--radius",
                    type=float,
                    default=None,
                    help="Manual KD radius")
    ap.add_argument("--kd_max",
                    type=int,
                    default=5,
                    help="# nearest samples for average")
    # ** Stylisation flags **
    ap.add_argument("--stylise",
                    choices=["none", "voxels"],
                    default="none",
                    help="Voxel stylisation")
    ap.add_argument("--voxel_size",
                    type=float,
                    default=0.05,
                    help="Voxel remesh size")
    ap.add_argument("--voxel_adaptivity",
                    type=float,
                    default=0.0,
                    help="Voxel remesh adaptivity")
    ap.add_argument("--keep_separate",
                    action="store_true",
                    help="Don't join meshes before remesh")
    # ** Colormap options **
    ap.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        help=
        "Colormap to use: blue, viridis, plasma, inferno, magma, turbo, coolwarm, seismic, rainbow, jet, etc."
    )
    ap.add_argument("--vmin",
                    type=float,
                    default=None,
                    help="Min value for colormap normalization")
    ap.add_argument("--vmax",
                    type=float,
                    default=None,
                    help="Max value for colormap normalization")
    ap.add_argument("--gamma",
                    type=float,
                    default=8.0,
                    help="Gamma/sharpness for sigmoid mapping (0 = linear)")
    ap.add_argument(
        "--feature",
        type=str,
        default="E",
        help="Feature to use for colormap",
        choices=["E", "density", "nu", "part_label", "material_id", "rgb", "clip_pca"])
    ap.add_argument(
        "--label_palette",
        type=str,
        default="default",
        choices=["default", "elasticity", "plasticity"],
        help=
        "Palette to use for part_label colouring (only applies when --feature part_label)."
    )
    ap.add_argument("--render_scene_scale",
                    nargs="+",
                    type=float,
                    default=[1.0])
    ap.add_argument("--transparent", action="store_true", default=False)
    ap.add_argument(
        "--is_dreamphysics",
        nargs="+",
        type=str2bool,
        default=[False],
        help="List of booleans, one per --obj_id, marking DreamPhysics assets",
    )
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--camera_id",
                    type=int,
                    default=None,
                    help="Camera index in transforms.json (0-based, optional)")
    ap.add_argument(
        "--blend",
        type=str2bool,
        default=False,
        help=
        "If true, blend original RGB shader with feature‐colour shader using an Empty-controlled gradient."
    )
    ap.add_argument(
        "--blend_feature",
        type=str,
        default="rgb",
        help="Secondary feature to blend with when --blend true.\n"
        "Use 'rgb' to blend with the original material (default).\n"
        "Otherwise choose another feature (E, density, nu, \n"
        "part_label, material_id, clip_pca) to blend two painted features.")
    ap.add_argument(
        "--save_blend",
        type=str2bool,
        default=False,
        help=
        "If true, also save the current Blender scene (.blend) for inspection."
    )
    ap.add_argument("--log_normalize_feature",
                    type=str2bool,
                    default=True,
                    help="If true, log-normalize the feature values.")
    ap.add_argument(
        "--material_types",
        type=str,
        default="plain,glossy",
        help="Comma-separated material styles for branch1 and branch2.\n"
        "Options per branch: plain | glossy | original.\n"
        "'plain' and 'glossy' create new Principled BSDFs with \n"
        "appropriate parameters; 'original' reuses the object's \n"
        "existing shader graph (only valid for branch2).")
    ap.add_argument(
        "--noise_edge",
        type=str2bool,
        default=False,
        help=
        "If true, add a Noise Texture modulation to the blend mask for a more organic edge."
    )
    ap.add_argument(
        "--noise_scale",
        type=float,
        default=50.0,
        help=
        "Noise texture scale when --noise_edge is true (higher = finer noise)."
    )
    ap.add_argument(
        "--noise_strength",
        type=float,
        default=0.3,
        help=
        "How strongly noise perturbs the edge (0-1). 0.3 gives subtle jitter; 1.0 full displacement."
    )
    ap.add_argument(
        "--focal_length",
        type=float,
        default=None,
        help="Camera focal length in mm. Overrides any camera intrinsics.")
    ap.add_argument(
        "--rotate_video",
        action="store_true",
        default=False,
        help=
        "If set, generate multiple frames for a 360° rotating video instead of a single still."
    )
    ap.add_argument("--views",
                    type=int,
                    default=120,
                    help="Number of views for 360° rotation (default: 120)")
    ap.add_argument("--data_dir",
                    type=str,
                    default=None,
                    help="Data directory containing the transforms.json file")
    ap.add_argument("--blend_file_path",
                    type=str,
                    default=None,
                    help="Path to the Blender scene file (.blend) to use for rendering")

    args = ap.parse_args(argv)

    # Pre-parse material_types into list so we don't repeat later
    mt = [s.strip().lower() for s in args.material_types.split(",")]
    # Ensure we have two values
    if len(mt) == 1:
        mt = mt * 2
    elif len(mt) > 2:
        mt = mt[:2]
    args.material_types = mt  # overwrite with list for convenience

    return args


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
#  SCENE UTILITIES
# -----------------------------------------------------------------------------


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in list(bpy.data.collections):
        if coll.users == 0 and coll.name != "Collection":
            bpy.data.collections.remove(coll)
    for db in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
               bpy.data.textures):
        for block in list(db):
            if block.users == 0:
                db.remove(block)


def scene_bbox(objs: List[bpy.types.Object]) -> Tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bb_min = Vector((1e9, 1e9, 1e9))
    bb_max = Vector((-1e9, -1e9, -1e9))
    for ob in objs:
        ob_eval = ob.evaluated_get(depsgraph)
        for corner in ob_eval.bound_box:
            wco = ob_eval.matrix_world @ Vector(corner)
            bb_min = Vector(map(min, bb_min, wco))
            bb_max = Vector(map(max, bb_max, wco))
    return bb_min, bb_max


def make_blended_material(mesh_objs,
                          layer_main: str = "CLIP_PCA",
                          layer_blend: str | None = None,
                          style_main: str = "plain",
                          style_blend: str = "plain",
                          *,
                          noise_edge: bool = False,
                          noise_scale: float = 50.0,
                          noise_strength: float = 0.3):
    """Create a material that blends between two vertex-colour layers along the Z-axis.

    Parameters
    ----------
    layer_main : str
        Name of the *first* vertex-colour layer.  This is the layer that will
        be visible where the gradient factor is 0 (i.e. towards the EMPTY).
    layer_blend : str | None
        Name of the *second* vertex-colour layer to blend to.  If *None* the
        original material of the object is used instead – this matches the
        previous behaviour of the function.  If provided, the original
        material is ignored and we blend *layer_main* → *layer_blend*.
    """

    # Re-use (or create) the empty that drives the gradient.
    if "BlendController" in bpy.data.objects:
        empty = bpy.data.objects["BlendController"]
    else:
        empty = bpy.data.objects.new("BlendController", None)
        bpy.context.collection.objects.link(empty)
        empty.location = (0, 0, 0.2
                          )  # Slightly above centre so original shows first

    for ob in mesh_objs:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            nt = mat.node_tree

            # Locate the output node
            out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")

            # ---- remove current surface link & remember old shader --------
            try:
                old_link = next(
                    l for l in nt.links
                    if l.to_node == out and l.to_socket.name == "Surface")
                old_shader = old_link.from_node
                nt.links.remove(old_link)
            except StopIteration:
                old_shader = None

            # Helper to create BSDF with style ---------------------------------
            def _create_bsdf(style: str, x: int, y: int):
                bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
                bsdf.location = (x, y)
                if style == "glossy":
                    bsdf.inputs["Roughness"].default_value = 0.0
                    bsdf.inputs["Specular IOR Level"].default_value = 1.0
                    bsdf.inputs["Metallic"].default_value = 0.1
                # "plain" keeps default values
                return bsdf

            # ---- FIRST feature branch (layer_main) -----------------------
            vc1 = nt.nodes.new("ShaderNodeVertexColor")
            vc1.layer_name = layer_main
            vc1.location = (-750, 250)
            bsdf1 = _create_bsdf(style_main, -500, 250)
            nt.links.new(vc1.outputs["Color"], bsdf1.inputs["Base Color"])

            # ---- Determine SECOND branch: either another layer or original shader
            if layer_blend is None:
                # Use original shader OR create new according to style_blend
                if style_blend == "original" and old_shader is not None:
                    branch2_shader = old_shader
                else:
                    branch2_shader = _create_bsdf(style_blend, -500, 0)
                    if old_shader is not None and style_blend == "plain":
                        # For plain we could adopt old base colour/texture by linking
                        # old_shader's BaseColor into new bsdf if it has one. Keep simple.
                        pass
            else:
                vc2 = nt.nodes.new("ShaderNodeVertexColor")
                vc2.layer_name = layer_blend
                vc2.location = (-750, 0)
                bsdf2 = _create_bsdf(style_blend, -500, 0)
                nt.links.new(vc2.outputs["Color"], bsdf2.inputs["Base Color"])
                branch2_shader = bsdf2

            # ---- gradient-mask branch ------------------------------------
            tc = nt.nodes.new("ShaderNodeTexCoord")
            tc.location = (-1000, -200)
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            sep.location = (-800, -200)
            ramp = nt.nodes.new("ShaderNodeMapRange")
            ramp.location = (-600, -200)
            # ramp.inputs["From Min"].default_value = -0.5
            # ramp.inputs["From Max"].default_value = 0.5
            ramp.inputs["From Min"].default_value = 0.5
            ramp.inputs["From Max"].default_value = -0.5
            # ramp.inputs["From Min"].default_value = -0.1 ## sharper transition!
            # ramp.inputs["From Max"].default_value = 0.1
            tc.object = empty
            nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
            nt.links.new(sep.outputs["Z"], ramp.inputs["Value"])

            # ---- optional noise modulation ------------------------------
            fac_output = ramp.outputs["Result"]  # default
            if noise_edge:
                noise = nt.nodes.new("ShaderNodeTexNoise")
                noise.location = (-900, -450)
                noise.inputs["Scale"].default_value = noise_scale
                noise.inputs["Roughness"].default_value = 0.0
                nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

                sub = nt.nodes.new("ShaderNodeMath")
                sub.operation = 'SUBTRACT'
                sub.location = (-700, -350)
                nt.links.new(noise.outputs["Fac"], sub.inputs[0])
                sub.inputs[1].default_value = 0.5  # center noise around 0

                mul = nt.nodes.new("ShaderNodeMath")
                mul.operation = 'MULTIPLY'
                mul.location = (-550, -350)
                nt.links.new(sub.outputs[0], mul.inputs[0])
                mul.inputs[1].default_value = noise_strength

                add = nt.nodes.new("ShaderNodeMath")
                add.operation = 'ADD'
                add.location = (-400, -350)
                nt.links.new(ramp.outputs["Result"], add.inputs[0])
                nt.links.new(mul.outputs[0], add.inputs[1])

                clamp = nt.nodes.new("ShaderNodeClamp")
                clamp.location = (-250, -350)
                nt.links.new(add.outputs[0], clamp.inputs[0])

                fac_output = clamp.outputs[0]

            # ---- final mix ------------------------------------------------
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.location = (-300, 50)
            # Socket order: shader1 (Fac=0) , shader2 (Fac=1)
            nt.links.new(bsdf1.outputs[0], mix.inputs[1])
            nt.links.new(branch2_shader.outputs[0], mix.inputs[2])
            nt.links.new(fac_output, mix.inputs["Fac"])
            nt.links.new(mix.outputs[0], out.inputs["Surface"])


# ── in normalise_objects() ──────────────────────────────────────────────
def normalize_objects(objs: list[bpy.types.Object], scene_scale: float = 1.0):
    """
    Translate the WHOLE hierarchy so its bbox centre is at the origin,
    then scale it so the largest side is 1 m (× scene_scale).
    Returns (scale, centre) so the same transform can be applied to the PLY.
    """
    bb_min, bb_max = scene_bbox(objs)
    centre = (bb_min + bb_max) / 2
    size_vec = bb_max - bb_min
    
    if max(size_vec) > 0:
        scale = scene_scale / max(size_vec)
    else:
        scale = scene_scale

    # Create the transformation matrices
    T = Matrix.Translation(-centre)
    S = Matrix.Scale(scale, 4)
    transform_matrix = S @ T
    
    # Apply the transformation to every object (not just root objects)
    for ob in objs:
        ob.matrix_world = transform_matrix @ ob.matrix_world
        
    # Force update
    bpy.context.view_layer.update()
    
    return float(scale), np.array(centre)


# -----------------------------------------------------------------------------
#  VOXEL STYLISATION
# -----------------------------------------------------------------------------

# def join_meshes(objs: List[bpy.types.Object]) -> bpy.types.Object:
#     if len(objs)==1:
#         ob = objs[0]
#     else:
#         bpy.ops.object.select_all(action="DESELECT")
#         for o in objs: o.select_set(True)
#         bpy.context.view_layer.objects.active = objs[0]
#         bpy.ops.object.join()
#         ob = bpy.context.view_layer.objects.active
#     bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
#     return ob


# Modify the join_meshes function to split leaves:
def join_meshes(objs: List[bpy.types.Object]):
    leaf_objs = [ob for ob in objs if "leafset" in ob.name.lower()]
    other_objs = [ob for ob in objs if ob not in leaf_objs]

    merged = []
    if other_objs:
        bpy.ops.object.select_all(action="DESELECT")
        for ob in other_objs:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = other_objs[0]
        bpy.ops.object.join()
        tgt = bpy.context.view_layer.objects.active
        bpy.ops.object.transform_apply(location=False,
                                       rotation=False,
                                       scale=True)
        merged.append(tgt)

    # apply scale to leaves separately
    for ob in leaf_objs:
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=False,
                                       rotation=False,
                                       scale=True)
        merged.append(ob)

    return merged




def voxel_remesh(ob: bpy.types.Object, size=0.02, adaptivity=0.0):
    print("Remeshing", ob.name)
    bpy.context.view_layer.objects.active = ob
    # mod = ob.modifiers.new("VoxelRemesh", type="REMESH")
    # mod.mode = "VOXEL"
    # mod.voxel_size = size
    # mod.adaptivity = adaptivity

    mod = ob.modifiers.new("BlocksRemesh", type="REMESH")
    if "leafset" in ob.name.lower():
        # mod.mode = "BLOCKS"
        # mod.octree_depth = 15
        mod.mode = "VOXEL"
        mod.voxel_size = 0.008
        mod.adaptivity = 0.0

    else:
        mod.mode = "BLOCKS"
        mod.octree_depth = 7
    # mod.scale = 0.9
    # mod.scale = 0.99
    # if "leafset" in ob.name.lower():
    #     return
    bpy.ops.object.modifier_apply(modifier=mod.name)


# -----------------------------------------------------------------------------
#  PLY & KD-TREE
# -----------------------------------------------------------------------------


def load_semantic_ply(ply_path, normalize=True, is_dreamphysics=False):
    """Load semantic colors and positions from the PLY file."""
    ply_data = PlyData.read(ply_path)
    vertex_elem = ply_data['vertex']  # PlyElement
    vertex_data = vertex_elem.data  # <-- structured NumPy array

    # positions
    positions = np.column_stack(
        (vertex_data['x'], vertex_data['y'], vertex_data['z']))

    features = {}
    # Discrete features (labels)
    if 'part_label' in vertex_data.dtype.names:
        features['part_label'] = vertex_data['part_label']
    if 'material_id' in vertex_data.dtype.names:
        features['material_id'] = vertex_data['material_id']
    # Continuous features
    if 'E' in vertex_data.dtype.names:
        features['E'] = vertex_data['E']
    if 'density' in vertex_data.dtype.names:
        features['density'] = vertex_data['density']
    if 'nu' in vertex_data.dtype.names:
        features['nu'] = vertex_data['nu']
    if is_dreamphysics:
        features['E'] *= 1e7
    if normalize:
        if 'E' in features:
            features['E'] = np.log(features['E'])
        if 'density' in features:
            features['density'] = np.log(features['density'])

    return positions, features


# -----------------------------------------------------------------------------
#  COLORMAP SYSTEM
# -----------------------------------------------------------------------------

# Built-in colormaps for when matplotlib is not available
BUILTIN_COLORMAPS = {
    'blue': [
        (0.02, 0.12, 0.24),  # Dark blue
        (0.00, 0.41, 0.71),  # Medium blue
        (0.00, 0.78, 1.00),  # Light cyan
    ],
    'viridis': [
        (0.267, 0.005, 0.329),  # Dark purple
        (0.128, 0.565, 0.551),  # Teal
        (0.993, 0.906, 0.144),  # Yellow
    ],
    'plasma': [
        (0.050, 0.030, 0.528),  # Dark blue
        (0.796, 0.280, 0.469),  # Pink
        (0.940, 0.975, 0.131),  # Yellow
    ],
    'inferno': [
        (0.001, 0.000, 0.014),  # Black
        (0.735, 0.215, 0.330),  # Red
        (0.988, 1.000, 0.645),  # Yellow-white
    ],
    'coolwarm': [
        (0.230, 0.299, 0.754),  # Blue
        (0.865, 0.865, 0.865),  # Gray
        (0.706, 0.016, 0.150),  # Red
    ],
    'turbo': [
        (0.190, 0.072, 0.583),  # Purple
        (0.100, 0.800, 0.300),  # Green
        (0.730, 0.150, 0.095),  # Red
    ],
}


def _lerp(a, b, t):
    """Linear interpolation between two colors."""
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _sigmoid(x, k=8.0):
    """Sigmoid function for non-linear mapping."""
    if k == 0:
        return x  # Linear mapping
    return 1 / (1 + np.exp(-k * (x - 0.5)))


def builtin_colormap(t_norm: float,
                     colormap_name: str,
                     gamma: float = 8.0) -> tuple[float, float, float]:
    """Apply a built-in colormap to a normalized value."""
    if colormap_name not in BUILTIN_COLORMAPS:
        print(f"Warning: Unknown colormap '{colormap_name}', using 'blue'")
        colormap_name = 'blue'

    colors = BUILTIN_COLORMAPS[colormap_name]
    t = _sigmoid(max(0, min(1, t_norm)), gamma)

    # Interpolate between colors
    n_colors = len(colors)
    if n_colors == 2:
        return _lerp(colors[0], colors[1], t)
    elif n_colors == 3:
        if t < 0.5:
            return _lerp(colors[0], colors[1], t / 0.5)
        else:
            return _lerp(colors[1], colors[2], (t - 0.5) / 0.5)
    else:
        # General case for more colors
        segment_size = 1.0 / (n_colors - 1)
        idx = int(t / segment_size)
        if idx >= n_colors - 1:
            return colors[-1]
        local_t = (t - idx * segment_size) / segment_size
        return _lerp(colors[idx], colors[idx + 1], local_t)


def load_clip_ply(ply_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return Nx3 positions and Nx3 RGB float colours (0‑1)."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.column_stack((v["x"], v["y"], v["z"]))
    if {"red", "green", "blue"}.issubset(v.dtype.names):
        rgb = np.column_stack((v["red"], v["green"], v["blue"])) / 255.0
    else:
        raise ValueError("PLY is missing colour fields (red/green/blue)")
    return xyz.astype(np.float32), rgb.astype(np.float32)


def build_kdtree(pts: np.ndarray) -> kdtree.KDTree:
    kd = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    return kd


def paint_mesh_with_clip(
    mesh_objs: List[bpy.types.Object],
    clip_pos: np.ndarray,
    clip_rgb: np.ndarray,
    radius: float | None = None,
    kd_max: int = 5,
    *,
    layer_name: str = "CLIP_PCA",
) -> None:
    """Fill *layer_name* vertex-colour layer with colours from CLIP PCA point-cloud."""

    # Build KD-tree once
    kd = build_kdtree(clip_pos)

    # Stats
    total_verts = 0
    mapped_verts = 0

    for ob in mesh_objs:
        me = ob.data
        if layer_name not in me.vertex_colors:
            me.vertex_colors.new(name=layer_name)
        vcol = me.vertex_colors[layer_name]

        # Ensure layer length matches loops
        if len(vcol.data) != len(me.loops):
            vcol.data.foreach_set("color",
                                  [0.0] * len(me.loops) * 4)  # RGBA zeros

        for poly in me.polygons:
            for li in poly.loop_indices:
                total_verts += 1
                vi = me.loops[li].vertex_index
                wco = ob.matrix_world @ me.vertices[vi].co

                nearest = kd.find((wco.x, wco.y, wco.z))  # (co, idx, dist)
                assert nearest is not None, "No nearest point found"
                if nearest is not None:
                    col = clip_rgb[nearest[1]]
                    vcol.data[li].color = (*col.tolist(), 1.0)
                    mapped_verts += 1

    # Print diagnostics
    print("\n[CLIP-PCA Colour Transfer]")
    print(f"  Point-cloud points : {len(clip_pos):6d}")
    print(f"  Number of vertices : {total_verts}")
    print(f"  Mesh loops         : {total_verts:6d}")
    print(
        f"  Loops coloured     : {mapped_verts:6d} ({mapped_verts/total_verts*100:5.1f} %)\n"
    )


def paint_mesh_with_feature(mesh_objs,
                            clip_pos,
                            E_vals,
                            kd_max=5,
                            radius=None,
                            vmin=None,
                            vmax=None,
                            colormap="plasma",
                            palette=None,
                            *,
                            layer_name: str = "CLIP_PCA"):
    # Print scene bounds for debugging
    print("\n[DEBUG] Scene bounds:")
    bb_min_mesh, bb_max_mesh = scene_bbox(mesh_objs)
    print(f"Mesh bounds: min={bb_min_mesh}, max={bb_max_mesh}")
    print(
        f"PLY bounds: min={clip_pos.min(axis=0)}, max={clip_pos.max(axis=0)}")

    # Build KD tree ----------------------------------------------------------
    kd = kdtree.KDTree(len(clip_pos))
    for i, p in enumerate(clip_pos):
        kd.insert(p, i)
    kd.balance()
    if radius is None:
        bb = clip_pos.ptp(0)  # diag of point cloud
        radius = 0.03 * np.linalg.norm(bb)
    print(f"[DEBUG] Using KD radius: {radius:.6f}")

    # Precompute colormap LUT if continuous
    lut = None
    if palette is None:
        cmap = cm.get_cmap(colormap)
        lut = (cmap(np.linspace(0, 1, 256))[:, :3]).astype(np.float32)
        # Ensure vmin and vmax are calculated based on the full E_vals if not provided
        # This was the original behavior and is generally correct for consistent coloring.
        current_vmin = np.min(E_vals) if vmin is None else vmin
        current_vmax = np.max(E_vals) if vmax is None else vmax
        scale = 255.0 / (current_vmax - current_vmin + 1e-12)

    # Track distance stats
    max_dist_overall = 0.0
    total_dist_overall = 0.0
    count_overall = 0
    skipped_overall = 0

    for ob in mesh_objs:
        me = ob.data
        if layer_name not in me.vertex_colors:
            me.vertex_colors.new(name=layer_name)
        vcol = me.vertex_colors[layer_name]

        num_verts = len(me.vertices)
        # Array to store the calculated feature value for each vertex of the current object
        vertex_feature_values = np.zeros(num_verts, dtype=np.float32)

        # --- Step 1: Calculate average feature value for each vertex ---
        for vert_idx, vert in enumerate(me.vertices):
            wco = ob.matrix_world @ vert.co
            hits = kd.find_n(
                wco, kd_max)  # kd.find_range(wco, radius) # Alternative

            if hits:
                dist = hits[0][2]  # Distance to nearest point
                max_dist_overall = max(max_dist_overall, dist)
                total_dist_overall += dist
                count_overall += 1

                # Optional: Skip if distance is too large, though this might leave uncolored vertices
                # if dist > radius * 2: # Example threshold
                #     skipped_overall += 1
                #     vertex_feature_values[vert_idx] = np.nan # Or some default value
                #     continue

                # Take feature values from E_vals based on indices from KD-tree hits
                feature_samples = [
                    E_vals[h[1]] for h in hits if h[1] < len(E_vals)
                ]
                if feature_samples:
                    vertex_feature_values[vert_idx] = np.mean(feature_samples)
                else:
                    # No valid hits (e.g. all hit indices out of bounds for E_vals)
                    # or no hits at all if find_range was used and found none.
                    vertex_feature_values[
                        vert_idx] = current_vmin if palette is None else 0  # Default to min or first label
            else:
                # No points found by KDTree (e.g. if find_range used and found none)
                skipped_overall += 1
                vertex_feature_values[
                    vert_idx] = current_vmin if palette is None else 0  # Default value

        # --- Step 2: Vectorized color calculation ---
        v_colors_rgb = np.zeros((num_verts, 3), dtype=np.float32)
        if palette is None:  # Continuous colormap
            if lut is not None:  # Should always be true if palette is None
                # Vectorized scaling and LUT lookup
                indices = np.clip(
                    ((vertex_feature_values - current_vmin) * scale), 0,
                    255).astype(np.int32)
                v_colors_rgb = lut[indices]
        else:  # Discrete palette (for part_label)
            # Ensure labels are integers for palette lookup
            int_labels = vertex_feature_values.astype(np.int32)

            # Handle potential out-of-bounds labels if palette is a list/array
            if isinstance(palette, dict):
                v_colors_rgb = np.array([
                    palette.get(lbl, (0, 0, 0)) for lbl in int_labels
                ])  # Default to black if label not in dict
            elif isinstance(palette, list):
                default_color_arr = np.array([0.0, 0.0,
                                              0.0])  # Default color (black)
                # Efficiently create colors array
                safe_labels = np.clip(int_labels, 0, len(palette) - 1)
                v_colors_rgb = np.array(palette)[safe_labels]
                # Identify out-of-bound original labels and set them to default (not strictly necessary with np.clip if that behavior is fine)
                # out_of_bounds_mask = (int_labels < 0) | (int_labels >= len(palette))
                # v_colors_rgb[out_of_bounds_mask] = default_color_arr
            else:  # Fallback for unexpected palette type
                v_colors_rgb = np.array(
                    [distinct_hsv_palette(10)[0]] *
                    num_verts)  # Default to first color of HSV palette

        # Combine with alpha channel
        v_colors_final = np.ones((num_verts, 4), dtype=np.float32)
        v_colors_final[:, :3] = v_colors_rgb

        # --- Step 3: Broadcast vertex → loops in one go ---
        flat_colors = np.empty(len(me.loops) * 4, dtype=np.float32)
        loop_vertex_indices = np.array(
            [loop.vertex_index for loop in me.loops])
        flat_colors = v_colors_final[loop_vertex_indices].ravel()
        vcol.data.foreach_set("color", flat_colors)

    # Print distance statistics (overall for all mesh objects processed)
    if count_overall > 0:
        avg_dist_overall = total_dist_overall / count_overall
        print(f"\n[DEBUG] Overall Distance statistics:")
        print(f"Max distance to nearest point: {max_dist_overall:.6f}")
        print(f"Average distance to nearest point: {avg_dist_overall:.6f}")
        print(f"Vertices skipped (no KD hits or too far): {skipped_overall}")
        print(f"Total vertices processed for KD search: {count_overall}")


def distinct_hsv_palette(n, seed=42):
    """Generate n visually-distinct colors that are vibrant but not oversaturated.
    The colors are similar to those in the reference image, but slightly darker
    for better contrast against a pure white background."""
    random.seed(seed)

    # Define a custom color palette inspired by the reference image
    # These colors work well under white lighting and white backgrounds
    # Each color is now slightly darker for better contrast
    base_colors = [
        (0.0, 0.64, 0.73),  # Darker Cyan
        (0.686, 0.765, 0.149),  # Specific Yellowish-Green (#afc326) from user
        (0.8, 0.25, 0.25),  # Darker Red
        (0.85, 0.4, 0.85),  # Darker Pink
        (0.212, 0.400, 0.221
         ),  # Specific Green from reference image (monster head) - Even Darker
        (0.15, 0.3, 0.85),  # Darker Blue
        (0.37, 0.17, 0.85),  # Darker Purple
        (0.8, 0.5, 0.0),  # Darker Orange
        (0.85, 0.68, 0.0),  # Darker Yellow
        (0.4, 0.75, 0.75),  # Darker Teal
        (0.68, 0.5, 0.86),  # Darker Lavender
        (0.78, 0.5, 0.3),  # Darker Coral
    ]
    return base_colors[:n]


def get_color_for_part_label(part_label: int,
                             num_part_label: int = 8,
                             *,
                             palette_type: str = 'default'):
    """Return an RGB colour (0-1) for the given *part_label*.

    If *palette_type* is 'default' the legacy *distinct_hsv_palette* is used.
    If 'elasticity' or 'plasticity', pre-defined warm/cool palettes are used
    so that the two groups cannot clash even if label IDs overlap.
    """
    if palette_type != 'default' and palette_type in PALETTE_MAP:
        palette = PALETTE_MAP[palette_type]
        if part_label >= len(palette):
            # Fallback to HSV palette if we run out of predefined colours
            palette = distinct_hsv_palette(num_part_label)
    else:
        palette = distinct_hsv_palette(num_part_label)

    assert part_label < len(
        palette
    ), f"Part label {part_label} is out of range for palette {palette_type} (len={len(palette)})"
    return palette[part_label]


def centre_and_scale(scene, mesh_objs, scene_scale=1.0):
    bb_min, bb_max = scene_bbox(mesh_objs)
    centre = (bb_min + bb_max) / 2
    size_vec = bb_max - bb_min
    scale = scene_scale / max(size_vec)  # longest edge → 1

    S = Matrix.Scale(scale, 4)  # uniform scale matrix

    for ob in scene.objects:
        if ob.parent is None:  # root nodes only
            mw = ob.matrix_world
            mw.translation = (mw.translation - centre) * scale
            ob.matrix_world = S @ mw  # scale rotations too

    bpy.context.view_layer.update()
    return float(scale), np.array(centre)


# -----------------------------------------------------------------------------
#  SHINY VERTEX-COLOUR MATERIAL
# -----------------------------------------------------------------------------


def make_glossy_vcol(name="ClipPCA_Glossy"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    vc = nt.nodes.new('ShaderNodeVertexColor')
    vc.layer_name = 'CLIP_PCA'
    vc.location = (-300, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (250, 0)
    nt.links.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = 1.0
    ## make more metalic
    bsdf.inputs["Metallic"].default_value = 0.1
    return m


def render_semantic_glb(semantic_glb_path,
                        output_dir,
                        render_scene_scale=1.0,
                        resolution=512,
                        transparent=False,
                        obj_id=None,
                        camera_id=None,
                        rotate_video=False,
                        focal_length=None,
                        views=120,
                        data_dir=None,
                        blend_file_path=None):
    """Generate a single image render of the semantic GLB using render_blender.py."""


    print(f"Rendering semantic GLB: {semantic_glb_path}")
    os.makedirs(f"{output_dir}/{semantic_glb_path.stem}", exist_ok=True)
    # Run render_blender.py with appropriate arguments
    render_cmd = f'blender -b -P pixie/blender/render_glb.py -- '
    render_cmd += f'--blend_file_path "{blend_file_path}" '
    render_cmd += f'--obj "{semantic_glb_path}" '
    render_cmd += f'--output_folder "{output_dir}/{semantic_glb_path.stem}" '
    num_views = views if rotate_video else 1
    render_cmd += f'--views {num_views} '
    render_cmd += f'--resolution {resolution} '
    render_cmd += f'--input_model glb '
    render_cmd += f'--scene_scale {render_scene_scale} '
    if transparent:
        render_cmd += f'--transparent '
    if rotate_video:
        render_cmd += '--rotate_video '
    if obj_id is not None:
        render_cmd += f'--obj_id {obj_id} '

    if camera_id is not None:
        render_cmd += f'--camera_id {camera_id} '

    if focal_length is not None:
        render_cmd += f'--focal_length {focal_length} '

    if data_dir is not None:
        render_cmd += f'--data_dir "{data_dir}" '

    print(f"Running render command: {render_cmd}")
    subprocess.run(render_cmd, shell=True, check=True)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------


def process_single_object(glb_path: Path,
                          pred_ply: Path,
                          clip_pred_ply: Path,
                          out_dir: Path,
                          args,
                          global_vmin: float = None,
                          global_vmax: float = None,
                          colormap_name: str = 'blue',
                          is_dreamphysics: bool = False,
                          render_scene_scale: float = 1.0,
                          obj_id: str = None) -> tuple[float, float]:
    """Process a single object and return its min/max E values."""
    print(f"\n[PROCESSING] Object: {glb_path}")

    # Check if output already exists
    colormap_suffix = f"_{args.colormap}" if args.colormap != "blue" else ""
    # out_glb = f"{glb_path.stem}_mat_{args.feature}_pred_voxelized{colormap_suffix}.glb"
    out_glb = f"{args.feature}.glb"
    if is_dreamphysics:
        out_glb = out_glb.replace(".glb", "_dreamphysics.glb")
    out_glb = out_dir / out_glb

    if out_glb.exists() and not args.overwrite:
        print(f"[SKIP] Output GLB already exists: {out_glb}")
        # Still do rendering
        render_semantic_glb(out_glb,
                            out_dir,
                            render_scene_scale=render_scene_scale,
                            transparent=args.transparent,
                            obj_id=obj_id,
                            camera_id=args.camera_id,
                            rotate_video=args.rotate_video,
                            focal_length=args.focal_length,
                            views=args.views,
                            data_dir=args.data_dir,
                            blend_file_path=args.blend_file_path)
        # Return dummy values since we didn't process the feature
        return 0.0, 1.0

    # Clear scene for this object
    clear_scene()

    # Import
    print(f"[IMPORT] {glb_path}")
    ext = glb_path.suffix.lower()
    if ext == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(glb_path), merge_vertices=True)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(glb_path))
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(glb_path))
    else:
        raise ValueError("Unsupported format")

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No mesh!")

    # Stylise voxels
    if args.stylise == "voxels":
        if not args.keep_separate:
            mesh_objs = join_meshes(mesh_objs)
        else:
            for m in mesh_objs:
                bpy.ops.object.transform_apply(location=False,
                                               rotation=False,
                                               scale=True)
        for m in mesh_objs:
            voxel_remesh(m, args.voxel_size, args.voxel_adaptivity)

    # Normalize mesh to unit cube if requested
    if args.normalize:
        bb_min_before, bb_max_before = scene_bbox(mesh_objs)
        print(f"\n[DEBUG] Mesh bounds BEFORE normalization: min={bb_min_before}, max={bb_max_before}, size={bb_max_before-bb_min_before}")
        print("[NORMALIZE] Normalizing mesh to unit cube...")
        scale_factor, centre = normalize_objects(mesh_objs, scene_scale=args.scene_scale)
        print(f"Normalized with scale_factor: {scale_factor}, centre: {centre}")
        # Update scene after normalization
        bpy.context.view_layer.update()
        bb_min_after, bb_max_after = scene_bbox(mesh_objs)
        print(f"[DEBUG] Mesh bounds AFTER normalization: min={bb_min_after}, max={bb_max_after}, size={bb_max_after-bb_min_after}\n")

    if args.feature != "rgb":
        # Load PLY + paint
        print(f"[PLY] {pred_ply}")
        if args.feature != "clip_pca":
            clip_pos, all_features = load_semantic_ply(
                pred_ply,
                normalize=args.log_normalize_feature,
                is_dreamphysics=is_dreamphysics)
            feature = all_features[args.feature]

        # If global min/max not provided, return local ones for global computation
        if global_vmin is None or global_vmax is None:
            print(f"Returning local min/max: {feature.min()}, {feature.max()}")
            return feature.min(), feature.max()

        palette = None
        if args.feature == "part_label":
            ## get all possible labels (not only the ones in the data)
            palette = {
                lbl:
                get_color_for_part_label(int(lbl),
                                         palette_type=args.label_palette)
                for lbl in range(8)
            }
        elif args.feature == "material_id":
            ## get all possible material IDs (not only the ones in the data)
            palette = {
                lbl:
                get_color_for_part_label(int(lbl),
                                         palette_type=args.label_palette)
                for lbl in range(8)
            }

        print(f"Palette: {palette}")
        # Paint with global values
        if args.feature == "clip_pca":
            clip_pos, clip_rgb = load_clip_ply(clip_pred_ply)
            kd = build_kdtree(clip_pos)
            paint_mesh_with_clip(mesh_objs,
                                 clip_pos,
                                 clip_rgb,
                                 radius=args.radius,
                                 kd_max=args.kd_max)
        else:
            paint_mesh_with_feature(
                mesh_objs,
                clip_pos,
                feature,
                radius=args.radius,
                kd_max=args.kd_max if args.feature not in ["part_label", "material_id"] else 1,
                colormap=colormap_name,
                vmin=global_vmin,
                vmax=global_vmax,
                palette=palette,
                layer_name="CLIP_PCA")

        # Handle blending --------------------------------------------------
        if args.blend:
            if args.blend_feature.lower() == "rgb":
                # Legacy behaviour – blend with original material
                make_blended_material(mesh_objs,
                                      layer_main="CLIP_PCA",
                                      layer_blend=None,
                                      style_main=args.material_types[0],
                                      style_blend=args.material_types[1],
                                      noise_edge=args.noise_edge,
                                      noise_scale=args.noise_scale,
                                      noise_strength=args.noise_strength)
            else:
                # -------------------------------------------------------
                # 1) Paint the SECOND feature onto a new vertex-colour layer
                # -------------------------------------------------------
                blend_layer_name = f"BLEND_{args.blend_feature.upper()}"

                if args.blend_feature == "clip_pca":
                    clip_pos2, clip_rgb2 = load_clip_ply(clip_pred_ply)
                    paint_mesh_with_clip(
                        mesh_objs,
                        clip_pos2,
                        clip_rgb2,
                        radius=args.radius,
                        kd_max=args.kd_max,
                        layer_name=blend_layer_name,
                    )
                else:
                    # Load semantic PLY if not already
                    # Re-use previously loaded clip_pos if available.
                    if args.feature != "clip_pca" and 'clip_pos' in locals():
                        clip_pos2 = clip_pos
                        all_features2 = all_features  # already loaded
                    else:
                        clip_pos2, all_features2 = load_semantic_ply(
                            pred_ply,
                            normalize=args.log_normalize_feature,
                            is_dreamphysics=is_dreamphysics,
                        )
                    if args.blend_feature not in all_features2:
                        raise ValueError(
                            f"blend_feature '{args.blend_feature}' not found in PLY {pred_ply}"
                        )
                    feature2 = all_features2[args.blend_feature]

                    palette2 = None
                    if args.blend_feature == "part_label":
                        palette2 = {
                            lbl:
                            get_color_for_part_label(
                                int(lbl), palette_type=args.label_palette)
                            for lbl in range(8)
                        }
                    elif args.blend_feature == "material_id":
                        palette2 = {
                            lbl:
                            get_color_for_part_label(
                                int(lbl), palette_type=args.label_palette)
                            for lbl in range(8)
                        }

                    paint_mesh_with_feature(
                        mesh_objs,
                        clip_pos2,
                        feature2,
                        radius=args.radius,
                        kd_max=args.kd_max
                        if args.blend_feature not in ["part_label", "material_id"] else 1,
                        colormap=colormap_name,
                        vmin=feature2.min(),
                        vmax=feature2.max(),
                        palette=palette2,
                        layer_name=blend_layer_name,
                    )

                # -------------------------------------------------------
                # 2) Create blended material between the TWO feature layers
                # -------------------------------------------------------
                make_blended_material(mesh_objs,
                                      layer_main="CLIP_PCA",
                                      layer_blend=blend_layer_name,
                                      style_main=args.material_types[0],
                                      style_blend=args.material_types[1],
                                      noise_edge=args.noise_edge,
                                      noise_scale=args.noise_scale,
                                      noise_strength=args.noise_strength)
        else:
            print("[MATERIAL] assigning glossy VCol…")
            # If no blending, we still respect material_types[0]
            if args.material_types[0] == "glossy":
                mat = make_glossy_vcol()
            else:
                mat = bpy.data.materials.new("PlainVCol")
                mat.use_nodes = True
                nt = mat.node_tree
                nt.nodes.clear()
                vc = nt.nodes.new('ShaderNodeVertexColor')
                vc.layer_name = 'CLIP_PCA'
                vc.location = (-200, 0)
                bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
                bsdf.location = (50, 0)
                nt.links.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
                out = nt.nodes.new('ShaderNodeOutputMaterial')
                out.location = (250, 0)
                nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
            for ob in mesh_objs:
                ob.data.materials.clear()
                ob.data.materials.append(mat)

    if args.feature == "rgb" or args.feature == "clip_pca":
        feature = np.array([0.0])  ## dummy
    # Export colored

    print(f"[EXPORT] {out_glb}")
    # When exporting the plain RGB asset we *must not* request the exporter to write
    # vertex-colour data – the original GLB already contains its own materials
    # (textures, vertex colours, etc.).  Forcing `export_vertex_color='MATERIAL'`
    # would make the exporter generate a new vertex-colour layer filled with the
    # default white colour, effectively wiping the original appearance.  We only
    # need this flag when we have explicitly painted a new colour layer (i.e.
    # for non-RGB features).

    export_vcol_flag = "MATERIAL" if args.feature != "rgb" else "NONE"

    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_materials="EXPORT",
        export_vertex_color=export_vcol_flag,
        export_normals=True,
        export_animations=False,
    )

    if args.save_blend:
        blend_path = out_dir / f"{glb_path.stem}_mat_{args.feature}{'_vox' if args.stylise=='voxels' else ''}.blend"
        print(f"[BLEND] saving scene → {blend_path}")
        bpy.ops.wm.save_as_mainfile(
            filepath=str(blend_path),
            check_existing=False,  # overwrite silently if --overwrite
            compress=False)  # set True if you prefer smaller files

    # Render if needed
    render_semantic_glb(out_glb,
                        out_dir,
                        render_scene_scale=render_scene_scale,
                        transparent=args.transparent,
                        obj_id=obj_id,
                        camera_id=args.camera_id,
                        rotate_video=args.rotate_video,
                        focal_length=args.focal_length,
                        views=args.views,
                        data_dir=args.data_dir,
                        blend_file_path=args.blend_file_path)

    return feature.min(), feature.max()


def main():
    args = parse_argv()
    out_dir = ensure_dir(args.output_dir)

    # Handle multiple objects
    if args.glb_paths:
        glb_paths = [Path(p) for p in args.glb_paths]
        if len(glb_paths) != len(args.obj_ids):
            raise ValueError(
                "Number of GLB paths must match number of object IDs")
    else:
        print(f"Fetching {len(args.obj_ids)} Objaverse assets...")
        mapping = objaverse.load_objects(uids=args.obj_ids)
        glb_paths = [Path(mapping[obj_id]) for obj_id in args.obj_ids]

    # Handle missing clip_pred_ply - if not provided, use None for each object
    if not args.clip_pred_ply:
        args.clip_pred_ply = [None] * len(args.obj_ids)

    # First pass: compute global min/max E values
    print("\n[PASS 1] Computing global E value range...")
    global_vmin = args.vmin
    global_vmax = args.vmax
    if args.vmin is None and args.vmax is None:
        global_vmin, global_vmax = float('inf'), float('-inf')
        print(f"args.obj_ids: {args.obj_ids}")
        print(f"glb_paths: {glb_paths}")
        print(f"args.pred_ply: {args.pred_ply}")
        print(f"args.clip_pred_ply: {args.clip_pred_ply}")
        print(f"args.is_dreamphysics: {args.is_dreamphysics}")
        print(f"args.render_scene_scale: {args.render_scene_scale}")
        assert len(args.obj_ids) == len(glb_paths) == len(args.pred_ply) == len(args.clip_pred_ply) == len(args.is_dreamphysics) == len(args.render_scene_scale), f"Number of objects, GLB paths, pred PLYs, clip pred PLYs, is dreamphysics, and render scene scales must match"
        for obj_id, glb_path, pred_ply_path, clip_pred_ply_path, is_dp, r_scale in zip(
                args.obj_ids, glb_paths, args.pred_ply, args.clip_pred_ply,
                args.is_dreamphysics, args.render_scene_scale):
            pred_ply = Path(pred_ply_path)
            print(f">>>>> Processing object: {obj_id}")
            local_min, local_max = process_single_object(
                glb_path,
                pred_ply,
                clip_pred_ply_path,
                out_dir,
                args,
                is_dreamphysics=is_dp,
                render_scene_scale=r_scale,
                obj_id=obj_id)
            global_vmin = min(global_vmin, local_min)
            global_vmax = max(global_vmax, local_max)

    print(
        f"\n[INFO] Global E value range: {global_vmin:.3f} to {global_vmax:.3f}"
    )

    # Second pass: apply consistent coloring
    print("\n[PASS 2] Applying consistent coloring across all objects...")
    assert len(args.obj_ids) == len(glb_paths) == len(args.pred_ply) == len(args.clip_pred_ply) == len(args.is_dreamphysics) == len(args.render_scene_scale), f"Number of objects, GLB paths, pred PLYs, clip pred PLYs, is dreamphysics, and render scene scales must match"
    for obj_id, glb_path, pred_ply_path, clip_pred_ply_path, is_dp, r_scale in zip(
            args.obj_ids, glb_paths, args.pred_ply, args.clip_pred_ply,
            args.is_dreamphysics, args.render_scene_scale):
        pred_ply = Path(pred_ply_path)
        print(f"Processing object: {obj_id}")
        process_single_object(glb_path,
                              pred_ply,
                              clip_pred_ply_path,
                              out_dir,
                              args,
                              global_vmin=global_vmin,
                              global_vmax=global_vmax,
                              colormap_name=args.colormap,
                              is_dreamphysics=is_dp,
                              render_scene_scale=r_scale,
                              obj_id=obj_id)

    print("\n[DONE] All objects processed successfully!")


if __name__ == "__main__":
    main()
