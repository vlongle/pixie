from functools import lru_cache
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float
from nerfstudio.fields.base_field import Field
from params_proto import PrefixProto, Proto
from pytorch3d.transforms import Transform3d

from f3rm_robot.assets import get_panda_gripper_mesh
from f3rm_robot.field_adapter import (
    FeatureFieldAdapter,
    get_alpha,
    ray_samples_from_coords,
)


class CollisionArgs(PrefixProto, cli_parse=False):
    """Arguments for collision checking a proposed grasp. The default values work well for the default Panda gripper."""

    alpha_threshold: float = Proto(0.2, help="Alpha threshold for a point to be considered to be occupied.")
    voxel_size: float = Proto(
        0.0075,
        help="Voxel size to voxelize the Panda gripper. You may need to adjust alpha if you change this.",
    )
    overlap_num: int = Proto(10, help="Number of overlapping points to be considered a collision.")

    allow_finger_collisions: bool = Proto(
        False,
        help="Whether to allow collisions between the fingers, and hence use the gripper model without the fingers.",
    )
    ray_samples_per_batch: int = Proto(
        2**22,
        help="Number of ray samples to use per batch for collision checking, decrease if running out of memory.",
    )


@lru_cache(maxsize=1)
def get_gripper_points() -> torch.Tensor:
    """
    Get the points (i.e., voxels) that the Panda gripper occupies. The mesh we load only includes the surface of the
    gripper, but this should be sufficient for coarse collision checking. You should do more sophisticated collision
    checking in your downstream motion planner.

    We cache this method as creating the voxel grid is a bit slow, and we can reuse the same points for all grasps.
    """
    # Load mesh and convert to voxel grid
    mesh = get_panda_gripper_mesh()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=CollisionArgs.voxel_size)

    # Get (x, y, z) coordinates of the voxels
    voxel_grid_indices = [v.grid_index for v in voxel_grid.get_voxels()]
    voxel_points = np.array([voxel_grid.get_voxel_center_coordinate(idx) for idx in voxel_grid_indices])

    if not CollisionArgs.allow_finger_collisions:
        # Remove points with z < 0.035 (i.e., points that are in the fingers)
        voxel_points = voxel_points[voxel_points[:, 2] >= 0.035]

    # Convert to tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voxel_points = torch.tensor(voxel_points).float().to(device)
    return voxel_points


def get_collision_info(
    field: Field, grasps_to_nerf: Transform3d
) -> Tuple[Bool[torch.Tensor, "num_grasps"], Float[torch.Tensor, "num_grasps num_points 3"]]:
    """
    Check if the gripper has a collision with the scene for the given grasps. Call this with torch.no_grad unless you
    need the gradients, otherwise it is much slower.

    Args:
        field: the NeRF field.
        grasps_to_nerf: the grasp transforms that take points in the gripper frame to NeRF coordinate frame.

    Returns:
        collision_detected: a boolean tensor of shape (N,) indicating whether a collision was detected for each grasp.
        gripper_points: the transformed gripper points in the NeRF coordinate frame for vis purposes.
    """
    # Transform gripper points by the grasps into the NeRF coordinate system
    og_gripper_points = get_gripper_points()
    gripper_points = grasps_to_nerf.transform_points(og_gripper_points)
    if gripper_points.ndim == 2:
        gripper_points = gripper_points.unsqueeze(0)
    gripper_points_flat = gripper_points.view(-1, 3)
    ray_samples = ray_samples_from_coords(gripper_points_flat)

    # Query the NeRF in batches to get density and compute alphas
    batch_size = CollisionArgs.ray_samples_per_batch
    density = []
    for i in range(0, len(ray_samples), batch_size):
        batch_ray_samples = ray_samples[i : i + batch_size]
        batch_density, _ = field.get_density(batch_ray_samples)
        density.append(batch_density)
    density = torch.cat(density, dim=0)

    # Compute alpha and check if overlap exceeds threshold to be considered a collision
    alpha_flat = get_alpha(density, CollisionArgs.voxel_size)
    alpha = alpha_flat.view(gripper_points.shape[:2])
    assert gripper_points.shape[:2] == alpha.shape[:2], "You messed up the shapes!"
    overlap_num = (alpha >= CollisionArgs.alpha_threshold).sum(dim=1)
    collision_detected = (overlap_num >= CollisionArgs.overlap_num).squeeze()
    return collision_detected, gripper_points


def has_collision(feature_field: FeatureFieldAdapter, grasps_to_world: Transform3d) -> Bool[torch.Tensor, "num_grasps"]:
    """
    Check if the gripper has a collision with the scene for the given grasps. Call this with torch.no_grad unless you
    need the gradients, otherwise it is much slower.

    Returns a boolean tensor for whether a collision was detected for each grasp
    """
    grasps_to_nerf = grasps_to_world.compose(feature_field.world_to_nerf)
    return get_collision_info(field=feature_field.rgb_field, grasps_to_nerf=grasps_to_nerf)[0]
