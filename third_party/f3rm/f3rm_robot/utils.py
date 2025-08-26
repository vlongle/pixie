from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float
from matplotlib import colormaps
from pytorch3d.transforms import Transform3d
from tqdm import tqdm

from f3rm_robot.load import LoadState


def get_heatmap(
    values: Float[Union[torch.Tensor, np.ndarray], "n"], cmap_name: str = "turbo", invert: bool = False,
    colormap_min=0.0, colormap_max=1.0,
) -> Float[np.ndarray, "n 3"]:
    """
    Get the RGB heatmap for a given set of values. We normalize the values to [0, 1] and then use the given
    colormap. We optionally invert the values before normalizing.

    Args:
        values: Values to convert to a heatmap.
        cmap_name: Name of the colormap to use.
        invert: Whether to invert the values before normalizing.
    Returns:
        RGB heatmap as a numpy array.
    """
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    values = values * (colormap_max - colormap_min) + colormap_min
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    rgb = colormaps[cmap_name](values)[..., :3]  # don't need alpha channel
    return rgb


def sample_point_cloud(
    load_state: LoadState,
    num_points: int,
    bbox_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bbox_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_bbox: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Sample a point cloud given the load state. We sample points until we have at least `num_points` points
    in the point cloud. We optionally use a bounding box to filter the points which is specified in the
    world frame (the load state contains the NeRF to world transform).

    Modified from: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/exporter/exporter_utils.py
    """
    pipeline = load_state.pipeline

    # Setup bounding box min and max
    comp_l = torch.tensor(bbox_min, device=pipeline.device)
    comp_m = torch.tensor(bbox_max, device=pipeline.device)
    assert torch.all(comp_l < comp_m), f"Bounding box min {bbox_min} must be smaller than max {bbox_max}"

    pbar = tqdm(total=num_points, desc="Sampling point cloud for visualization")
    points = []
    rgbs = []

    while pbar.n < num_points:
        with torch.no_grad():
            ray_bundle, _ = pipeline.datamanager.next_train(0)
            output = pipeline.model(ray_bundle)
        rgb = output["rgb"]

        # Convert depth to NeRF then world coordinates
        depth = output["depth"]
        nerf_points = ray_bundle.origins + ray_bundle.directions * depth
        world_points = load_state.nerf_to_world.transform_points(nerf_points)

        # Only keep points within the bounding box
        if use_bbox:
            mask = torch.all(torch.concat([world_points > comp_l, world_points < comp_m], dim=-1), dim=-1)
            world_points = world_points[mask]
            rgb = rgb[mask]

        points.append(world_points)
        rgbs.append(rgb)
        pbar.update(len(world_points))

    # Concat all points and rgbs and create a point cloud
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    return pcd


@lru_cache(maxsize=1)
def get_gripper_mesh(include_sphere: bool = True, radius: float = 0.003) -> o3d.geometry.TriangleMesh:
    """Get a skeleton gripper mesh."""
    # Create left and right fingers
    left_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.075)
    right_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.075)

    # Offset between bottom of fingers and the center of the grasp
    left_finger.translate((0, 0.04, 0.075 / 2 - 0.01))
    right_finger.translate((0, -0.04, 0.075 / 2 - 0.01))

    # Bar connecting the fingers, rotate so it's horizontal
    bar = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.084)
    bar.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    bar.translate((0, 0, 0.065))

    # Top extension of the gripper
    top_ext = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.08)
    top_ext.translate((0, 0, 0.105))

    if include_sphere:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        gripper_mesh = left_finger + right_finger + top_ext + bar + sphere
    else:
        gripper_mesh = left_finger + right_finger + top_ext + bar
    return gripper_mesh


def get_gripper_meshes(gripper_poses: Transform3d) -> Tuple[Float[np.ndarray, "n v 3"], Float[np.ndarray, "n f 3"]]:
    """
    Get vertices and faces for given gripper poses. Used for visualization purposes and returns the vertices and faces
    for each gripper pose.
    """
    # Get the gripper mesh and transform the vertices by each gripper pose
    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices = torch.from_numpy(vertices).float().to(gripper_poses.device)
    with torch.no_grad():
        all_vertices = gripper_poses.transform_points(vertices)

    # Get the faces
    faces = np.asarray(gripper_mesh.triangles)
    faces = torch.from_numpy(faces).to(gripper_poses.device)
    all_faces = faces.repeat(len(gripper_poses), 1, 1)

    # Convert to numpy
    all_vertices = all_vertices.cpu().numpy()
    all_faces = all_faces.cpu().numpy()
    return all_vertices, all_faces
