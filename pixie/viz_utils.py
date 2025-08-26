import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
from plyfile import PlyData
import random
from pathlib import Path
from pixie.utils import run_cmd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import json


def viz_cuboid(ax, center, size, color='r', linewidth=1, linestyle='-', alpha=1.0):
    """
    Plots a 3D wireframe cuboid given center and size.

    Parameters
    ----------
    ax : matplotlib Axes3D
    center : list or array [x, y, z]
        Center coordinates of the cuboid
    size : list or array [dx, dy, dz]
        Half-sizes of the cuboid in each dimension
    color : color string or RGB tuple
    linewidth : float
    linestyle : string
    alpha : float
    """
    x_c, y_c, z_c = center
    dx, dy, dz = size
    
    # Calculate the 8 vertices of the cuboid
    x_min, x_max = x_c - dx, x_c + dx
    y_min, y_max = y_c - dy, y_c + dy
    z_min, z_max = z_c - dz, z_c + dz
    
    # Define the 12 edges of the cuboid
    edges = [
        # Bottom face
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
        # Top face
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
        # Vertical edges
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max])
    ]
    
    # Plot each edge
    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def viz_cuboid_plotly(fig, center, size, color='red', linewidth=2, opacity=1.0, name=None, rotation=None):
    """
    Adds a 3D wireframe cuboid to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or None
        Existing figure to add cuboid to. If None, creates new figure.
    center : list or array [x, y, z]
        Center coordinates of the cuboid
    size : list or array [dx, dy, dz]
        Half-sizes of the cuboid in each dimension
    color : color string or RGB tuple
        Color of the cuboid edges
    linewidth : float
        Width of the cuboid edges
    opacity : float
        Opacity of the cuboid (0-1)
    name : str, optional
        Name for the cuboid (appears in legend if provided)
    rotation : None or length-3 list/tuple of floats
        Euler angles in degrees (rot_x, rot_y, rot_z).  If provided, the
        cuboid is rotated by rot_x about the X-axis, then rot_y about Y, then
        rot_z about Z (all in degrees) before plotting.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure with the cuboid added
    """
    x_c, y_c, z_c = center
    dx, dy, dz = size
    
    # Calculate the 8 vertices of the cuboid
    x_min, x_max = x_c - dx, x_c + dx
    y_min, y_max = y_c - dy, y_c + dy
    z_min, z_max = z_c - dz, z_c + dz
    
    # Define vertices
    vertices = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max],  # 7
    ])
    
    # --- Apply rotation if requested ---------------------------------------
    if rotation is not None:
        # Expect rotation = [rot_x, rot_y, rot_z], in degrees
        rot = np.asarray(rotation, dtype=float)
        if rot.shape != (3,):
            raise ValueError("`rotation` must be a length‐3 iterable: [rot_x, rot_y, rot_z] in degrees.")

        # Convert degrees → radians
        rx, ry, rz = np.deg2rad(rot)

        # Rotation matrix around X-axis
        Rx = np.array([
            [1,          0,           0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])

        # Rotation matrix around Y-axis
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [          0, 1,          0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation matrix around Z-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [         0,           0, 1]
        ])

        # Combined rotation: first X, then Y, then Z
        R = Rz @ Ry @ Rx
        
        # Translate to origin, rotate, then translate back
        center_array = np.array(center)
        vertices = vertices - center_array  # Translate to origin
        vertices = vertices @ R.T            # Apply rotation
        vertices = vertices + center_array   # Translate back
    # ------------------------------------------------------------------------
    
    # Define edges as pairs of vertex indices
    edges = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Create edge traces
    edge_trace = []
    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0], None]
        y = [vertices[edge[0]][1], vertices[edge[1]][1], None]
        z = [vertices[edge[0]][2], vertices[edge[1]][2], None]
        edge_trace.extend([x, y, z])
    
    # Flatten the coordinates
    x_edges = []
    y_edges = []
    z_edges = []
    for i in range(0, len(edge_trace), 3):
        x_edges.extend(edge_trace[i])
        y_edges.extend(edge_trace[i+1])
        z_edges.extend(edge_trace[i+2])
    
    # Create the 3D line trace for all edges
    cuboid_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color=color, width=linewidth),
        opacity=opacity,
        name=name,
        showlegend=bool(name)  # Only show in legend if name is provided
    )
    
    # Handle figure creation or addition
    if fig is None:
        fig = go.Figure(data=[cuboid_trace])
        fig.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=0, z=0)),
                aspectmode="data",
                dragmode="orbit",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )
    else:
        fig.add_trace(cuboid_trace)
    
    return fig


def viz_plotly(
        points,
        colors,
        *,
        discrete_feature=False,
        continuous_feature=False,
        cmap="turbo",          # Plotly's built-in name; can also be a list/tuple
        vmin=None,
        vmax=None,
        marker_size=2,
        opacity=0.8,
        rotation=None,           # <-- New: [rot_x, rot_y, rot_z] in degrees
        fig=None,               # <-- New: existing figure to add trace to
        show=True,              # <-- New: whether to show the figure
        name=None,              # <-- New: trace name for legend
    ):
    """
    3-D scatter with Plotly, with an optional rotation of the point cloud.

    Parameters
    ----------
    points : (N,3) array-like
    colors : array-like –  1-D values or RGB triplets
    discrete : bool       – treat `colors` as class labels
    continuous : bool     – treat `colors` as scalars
    cmap : str | list     – Plotly colorscale (only used if `continuous`)
    vmin, vmax : float    – explicit colour-range limits (continuous) or
                            int limits (discrete).  If None they are inferred.
    marker_size : int/float
    opacity : float
    rotation : None or length-3 list/tuple of floats
        Euler angles in degrees (rot_x, rot_y, rot_z).  If provided, each
        point is rotated by rot_x about the X-axis, then rot_y about Y, then
        rot_z about Z (all in degrees) before plotting.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add trace to. If None, creates new figure.
    show : bool
        Whether to display the figure. Default True.
    name : str, optional
        Name for the trace (appears in legend if provided).
    """
    points = np.asarray(points)
    colors = np.asarray(colors)

    # --- Apply rotation if requested ---------------------------------------
    if rotation is not None:
        # Expect rotation = [rot_x, rot_y, rot_z], in degrees
        rot = np.asarray(rotation, dtype=float)
        if rot.shape != (3,):
            raise ValueError("`rotation` must be a length‐3 iterable: [rot_x, rot_y, rot_z] in degrees.")

        # Convert degrees → radians
        rx, ry, rz = np.deg2rad(rot)

        # Rotation matrix around X-axis
        Rx = np.array([
            [1,          0,           0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])

        # Rotation matrix around Y-axis
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [          0, 1,          0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation matrix around Z-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [         0,           0, 1]
        ])

        # Combined rotation: first X, then Y, then Z
        R = Rz @ Ry @ Rx
        # Apply to all points
        points = points @ R.T
    # ------------------------------------------------------------------------

    if discrete_feature:
        colors_int = colors.astype(int)
        if vmin is None:
            vmin = int(colors_int.min())
        if vmax is None:
            vmax = int(colors_int.max())

        color_discrete_map = px.colors.qualitative.Plotly
        # Build an evenly-spaced colourscale for integers
        colorscale = [
            [(i / max(1, len(color_discrete_map) - 1)), col]
            for i, col in enumerate(color_discrete_map)
        ]

        marker_kw = dict(
            size=marker_size,
            color=colors_int,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title="Class ID"),
            opacity=opacity,
        )

    elif continuous_feature:
        if vmin is None:
            vmin = float(colors.min())
        if vmax is None:
            vmax = float(colors.max())

        marker_kw = dict(
            size=marker_size,
            color=colors,
            colorscale=cmap,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title="Value"),
            opacity=opacity,
        )

    else:  # treat `colors` as literal RGB or named colours (no colourbar)
        marker_kw = dict(size=marker_size, color=colors, opacity=opacity)

    # Create scatter trace
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker_kw,
        name=name,
    )

    # Handle figure creation or addition
    if fig is None:
        fig = go.Figure(data=[scatter])
        fig.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=0, z=0)),
                aspectmode="data",
                dragmode="orbit",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )
    else:
        fig.add_trace(scatter)

    if show:
        fig.show()
    
    return fig



def distinct_hsv_palette(n: int, seed: int = 42):
    """Return *n* well-spaced RGB colours (0-1 floats)."""
    random.seed(seed)
    base = [
        (0.00, 0.64, 0.73),  # cyan-ish
        (0.686, 0.765, 0.149),  # lime
        (0.80, 0.25, 0.25),  # red
        (0.85, 0.40, 0.85),  # magenta
        (0.212, 0.400, 0.221),  # dark-green
        (0.15, 0.30, 0.85),  # blue
        (0.37, 0.17, 0.85),  # purple
        (0.80, 0.50, 0.00),  # orange
        (0.85, 0.68, 0.00),  # yellow
        (0.40, 0.75, 0.75),  # teal
        (0.68, 0.50, 0.86),  # lavender
        (0.78, 0.50, 0.30),  # coral
    ]
    # recycle if n > len(base)
    return (base * ((n + len(base) - 1) // len(base)))[:n]


PALETTE_MAP = {}  # hook for future warm-/cool palettes etc.


def get_color_for_part_label(part_label: int,
                             num_part_label: int = 8,
                             *,
                             palette_type: str = 'default'):
    """Return an RGB colour (0-1) for *part_label* using a fixed palette."""
    if palette_type != 'default' and palette_type in PALETTE_MAP:
        palette = PALETTE_MAP[palette_type]
        if part_label >= len(palette):  # fall back if palette runs short
            palette = distinct_hsv_palette(num_part_label)
    else:
        palette = distinct_hsv_palette(num_part_label)

    # ------------------------------------------------------------------
    # Handle "invalid" or sentinel labels (e.g. -1 from nearest-neighbour
    # mapping when no match is found).  We show them in a neutral grey so
    # they are still visible but clearly distinguished from valid labels.
    # ------------------------------------------------------------------
    if part_label < 0:
        return (0.6, 0.6, 0.6)  # light grey (0-1 range)

    if part_label >= len(palette):
        raise ValueError(
            f"Part label {part_label} ≥ palette length {len(palette)}")
    return palette[part_label]


def _prep_colors(c,
                 *,
                 cmap='viridis',
                 vmin=None,
                 vmax=None,
                 discrete=False,
                 palette_type='default'):
    """
    Accepts
        • Nx3 RGB                     → returns unchanged RGB
        • N scalar floats/int         → continuous colormap
        • N integer labels (discrete) → palette colours + legend helpers
    Returns
        rgb (N,3), is_scalar, norm, legend_info
        legend_info is None for continuous; otherwise (uniq_labels, colour_list)
    """
    c = np.asarray(c)

    # ---------- discrete labels ---------- #
    if discrete:
        if c.ndim != 1:
            raise ValueError(
                "Discrete mode expects a 1-D array of integer labels.")
        labels = c.astype(int)
        uniq = np.unique(labels)
        # palette size = max label idx + 1  (robust if labels not 0…K−1)
        rgb = np.array([
            get_color_for_part_label(lbl,
                                     num_part_label=int(labels.max()) + 1,
                                     palette_type=palette_type)
            for lbl in labels
        ],
                       dtype=float)
        legend_info = (uniq, [
            get_color_for_part_label(lbl,
                                     num_part_label=int(labels.max()) + 1,
                                     palette_type=palette_type) for lbl in uniq
        ])
        return rgb, False, None, legend_info

    # ---------- continuous scalars ---------- #
    if c.ndim == 1:
        vmin = c.min() if vmin is None else vmin
        vmax = c.max() if vmax is None else vmax
        norm = mcolors.Normalize(vmin, vmax, clip=True)
        rgb = cm.get_cmap(cmap)(norm(c))[:, :3]
        return rgb, True, norm, None

    # ---------- already RGB ---------- #
    if c.max() > 1.0:  # assume 0-255 ints
        c = c / 255.0
    return c, False, None, None



# ───────────────────────────────────────────────────────────
#  3-D scatter with scalar colour-bar  OR  label legend
# ───────────────────────────────────────────────────────────
def viz(
    points,
    colors=None,
    ax=None,
    *,
    elev=0,
    azim=0,
    size=1,
    cmap="turbo",
    title=None,
    vmin=None,
    vmax=None,
    discrete=False,
    palette_type="default",
    show_colorbar=True,
    cbar_kwargs=None,
    label_names=None,
    scene_bounds=None,
    rotation=None,
):
    """
    3-D scatter that automatically adds:

      • a colour-bar (continuous scalar data), or
      • a legend   (discrete labels).

    Parameters
    ----------
    points : (N,3) float
    colors : (N,3) RGB | (N,) scalars | (N,) int labels
    discrete       : True → treat *colors* as integer part labels
    palette_type   : choose alternative palettes (future-proof)
    label_names    : optional list/array mapping label-id → text in legend
    rotation : None or length-3 list/tuple of floats
        Euler angles in degrees (rot_x, rot_y, rot_z).  If provided, each
        point is rotated by rot_x about the X-axis, then rot_y about Y, then
        rot_z about Z (all in degrees) before plotting.
    """

    # --- figure / axis boilerplate -----------------------------------------
    created_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    # ------------------------------------------------------------------
    # Optional clipping to an axis-aligned bounding box (scene_bounds)
    # scene_bounds should be ((xmin, ymin, zmin), (xmax, ymax, zmax)).
    # Points outside are discarded from the plot.  Axis limits are set so
    # the resulting view is nicely framed to that box.
    # ------------------------------------------------------------------

    pts = np.asarray(points)
    if colors is None:
        cols = np.full((len(points), 3), 0.5)  # Gray color for all points
    else:
        cols = colors
    
    # --- Apply rotation if requested ---------------------------------------
    if rotation is not None:
        # Expect rotation = [rot_x, rot_y, rot_z], in degrees
        rot = np.asarray(rotation, dtype=float)
        if rot.shape != (3,):
            raise ValueError("`rotation` must be a length‐3 iterable: [rot_x, rot_y, rot_z] in degrees.")

        # Convert degrees → radians
        rx, ry, rz = np.deg2rad(rot)

        # Rotation matrix around X-axis
        Rx = np.array([
            [1,          0,           0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])

        # Rotation matrix around Y-axis
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [          0, 1,          0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation matrix around Z-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [         0,           0, 1]
        ])

        # Combined rotation: first X, then Y, then Z
        R = Rz @ Ry @ Rx
        # Apply to all points
        pts = pts @ R.T
    # ------------------------------------------------------------------------
    
    if scene_bounds is not None:
        bmin = np.asarray(scene_bounds[0])
        bmax = np.asarray(scene_bounds[1])
        msk = np.all((pts >= bmin) & (pts <= bmax), axis=1)
        if msk.any():
            pts = pts[msk]
            cols = cols[msk]

    # --- colour handling ----------------------------------------------------
    rgb, is_scalar, norm, legend_info = _prep_colors(
        cols,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        discrete=discrete,
        palette_type=palette_type,
    )

    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=rgb)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)

    # --- continuous colour-bar ---------------------------------------------
    if is_scalar and show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        kw = dict(shrink=0.75, pad=0.02)
        if cbar_kwargs:
            kw.update(cbar_kwargs)
        fig.colorbar(sm, ax=ax, **kw)

    # --- discrete legend ----------------------------------------------------
    if discrete and legend_info is not None:
        uniq_labels, uniq_cols = legend_info
        handles = [Patch(facecolor=col, edgecolor="k") for col in uniq_cols]
        if label_names is not None:
            texts = [label_names[int(i)] for i in uniq_labels]
        else:
            texts = [str(int(i)) for i in uniq_labels]
        # Keep legend tight to the plot; feel free to tweak loc/fontsize
        ax.legend(
            handles,
            texts,
            title="Part label",
            loc="upper center",
            ncols=3,
              bbox_to_anchor=(0.5, 1.0),
            borderaxespad=0.2,
            frameon=False,
        )

    # Apply axis limits if scene_bounds provided
    if scene_bounds is not None:
        ax.set_xlim(bmin[0], bmax[0])
        ax.set_ylim(bmin[1], bmax[1])
        ax.set_zlim(bmin[2], bmax[2])

    return ax


def viz_pred(pred, mask, ax=None):
    """
    pred is a numpy array of shape (11, D, D, D)
    """
    seg = pred[3:, :]
    labels = seg.argmax(axis=0)    # (64, 64, 64)

    # 2) Convert mask to boolean and keep only voxels the mask says are valid
    mask_bool = mask.astype(bool)  # True = plot it, False = ignore it

    # 3) Coordinates of masked voxels
    xs, ys, zs = np.nonzero(mask_bool)        # each is 1‑D and has the same length
    colors     = labels[mask_bool]            # colour = class‑id of each voxel
    pos = np.stack([xs, ys, zs], axis=-1)

    viz(pos, colors, ax=ax,
    discrete=True)


def load_semantic_ply(ply_path, normalize=True, is_dreamphysics=False):
    """Load semantic colors and positions from the PLY file."""
    ply_data = PlyData.read(ply_path)
    vertex_elem = ply_data["vertex"]  # PlyElement
    vertex_data = vertex_elem.data  # <-- structured NumPy array

    # positions
    positions = np.column_stack(
        (vertex_data["x"], vertex_data["y"], vertex_data["z"]))

    features = {}
    if "part_label" in vertex_data.dtype.names:
        features["part_label"] = vertex_data["part_label"]
    if "E" in vertex_data.dtype.names:
        features["E"] = vertex_data["E"]
    if "density" in vertex_data.dtype.names:
        features["density"] = vertex_data["density"]
    if "nu" in vertex_data.dtype.names:
        features["nu"] = vertex_data["nu"]
    if "material_id" in vertex_data.dtype.names:
        features["material_id"] = vertex_data["material_id"]

    nan_mask = np.isnan(positions).any(axis=1)
    if np.any(nan_mask):
        num_nan = np.sum(nan_mask)
        total = len(positions)
        print(
            f"⚠️ Warning: Found {num_nan}/{total} points with NaN coordinates in {ply_path}. Removing them."
        )

        positions = positions[~nan_mask]
        for key in features:
            features[key] = features[key][~nan_mask]

    if is_dreamphysics:
        if "E" in features:
            features["E"] *= 1e7
    if normalize:
        if "E" in features:
            features["E"] = np.log(features["E"])
        if "density" in features:
            features["density"] = np.log(features["density"])

    return positions, features


def compile_video(frames_dir: Path, out_mp4: Path, fps: float ):
    """Stitches PNG frames into an MP4 video using ffmpeg."""
    txt = frames_dir / "inputs.txt"
    # Ensure frames are sorted correctly, assuming 'frame_xxxxx.png' format
    frames = sorted(frames_dir.glob('*.png'), key=lambda p: int(p.stem.split('_')[-1]))
    txt.write_text("\n".join([f"file '{f.name}'" for f in frames]))
    ffmpeg_cmd = "ffmpeg" 
    cmd = (
        f"{ffmpeg_cmd} -y -r {fps} -f concat -safe 0 -i {txt} "
        f"-c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 "
        f"'{out_mp4}'"  # Use quotes for safety
    )
    print(f"🎞️  Creating video: {cmd}")
    run_cmd(cmd, step_name=f"compile_video {out_mp4}")