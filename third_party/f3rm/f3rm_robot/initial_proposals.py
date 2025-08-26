"""
Utilities for generating initial proposals.
"""
from typing import Tuple

import numpy as np
import torch
from jaxtyping import Float
from pytorch3d.ops import knn_points


class NoProposalsError(Exception):
    """Error raised when there are no proposals due to the masking."""

    pass


def dense_voxel_grid(
    min_bounds: Tuple[float, float, float], max_bounds: Tuple[float, float, float], voxel_size: float
) -> Float[torch.Tensor, "n_x n_y n_z 3"]:
    """Create a dense voxel grid between min and max bounds with given voxel size."""
    voxel_grid = torch.meshgrid(
        *(torch.arange(min_bound, max_bound, voxel_size) for min_bound, max_bound in zip(min_bounds, max_bounds)),
        indexing="ij",
    )
    voxel_grid = torch.stack(voxel_grid, dim=-1)
    return voxel_grid


def otsu_mask(voxel_sims: Float[torch.Tensor, "n"], num_bins: int = 100) -> Tuple[Float[torch.Tensor, "n"], float]:
    """
    Compute mask based on Otsu's method - i.e., maximize between-class variance.
    Returns the mask and threshold value.
    """
    # Normalize the similarities
    sim_min, sim_max = voxel_sims.min(), voxel_sims.max()
    voxel_sims_norm = (voxel_sims - sim_min) / (sim_max - sim_min)

    # Compute the histogram of the similarities
    threshold_vals = torch.linspace(0, 1, num_bins).to(voxel_sims.device)
    hist = torch.histc(voxel_sims_norm, bins=num_bins, min=0, max=1)
    probs = hist / torch.sum(hist)

    # Compute cumulative sum and mean
    cum_sum = torch.cumsum(probs, dim=0)
    cum_mean = torch.cumsum(probs * threshold_vals, dim=0)

    # Calculate the between-class variance for all possible threshold values
    variance = ((cum_mean[-1] * cum_sum - cum_mean) ** 2) / (cum_sum * (1 - cum_sum) + 1e-9)

    # Find threshold that maximizes the between-class variance
    threshold = threshold_vals[torch.argmax(variance)]
    mask = (voxel_sims_norm >= threshold).squeeze()
    return mask, threshold


def marching_cubes_mask(
    alpha: Float[torch.Tensor, "n_x n_y n_z"],
    iso_value: float,
    min_bounds: Tuple[float, float, float],
    max_bounds: Tuple[float, float, float],
) -> Float[torch.Tensor, "n 3"]:
    """
    Use marching cubes, so we only extract the surface of a voxel grid to reduce the number of proposals.
    Expects alpha to be a 3D tensor, not flattened.

    Returns:
        alpha_voxel_grid: the vertices of the mesh, **flattened** as a (n, 3) tensor.
    """
    from mcubes import marching_cubes
    from trimesh import Trimesh

    assert alpha.dim() == 3, f"alpha should be 3D, not {alpha.shape}"

    # We use mcubes as it's sufficiently fast (<0.02s). pytorch3d and kaolin are painful to deal with.
    verts, faces = marching_cubes(alpha.cpu().numpy(), iso_value)

    # Create a Trimesh, then scale and translate back to the original world coordinates
    mesh = Trimesh(verts, faces)
    diff_bounds = np.array(max_bounds) - np.array(min_bounds)
    scale = diff_bounds / np.array(alpha.shape)
    assert (scale > 0).all(), f"scale should be all positive, not {scale}"
    mesh.apply_scale(scale)
    mesh.apply_translation(min_bounds)

    # Use the vertices of the mesh as the new voxel grid
    alpha_voxel_grid = torch.from_numpy(mesh.vertices).float()
    alpha_voxel_grid = alpha_voxel_grid.to(alpha.device)
    return alpha_voxel_grid


def voxel_downsample(voxel_grid: Float[torch.Tensor, "n 3"], voxel_size: float) -> Float[torch.Tensor, "n 3"]:
    """Downsample a voxel grid by taking mean position of each voxel."""
    # Based on https://github.com/isl-org/Open3D/blob/master/cpp/open3d/geometry/PointCloud.cpp#L354
    voxel_min_bound = voxel_grid.amin(dim=0) - voxel_size * 0.5
    ref_coords = (voxel_grid - voxel_min_bound) / voxel_size
    voxel_indices = torch.floor(ref_coords).long()
    voxel_indices, voxel_indices_inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)

    # Tensor to hold sum and counts
    voxel_sum = torch.zeros(
        (voxel_indices.shape[0], voxel_grid.shape[1]), dtype=voxel_grid.dtype, device=voxel_grid.device
    )
    voxel_count = torch.zeros((voxel_indices.shape[0],), dtype=torch.int64, device=voxel_grid.device)

    # Add the voxel values to the voxel_sum tensor and increment the voxel_count tensor
    voxel_sum.scatter_add_(0, voxel_indices_inverse.unsqueeze(1).repeat(1, voxel_grid.shape[1]), voxel_grid)
    voxel_count.scatter_add_(0, voxel_indices_inverse, torch.ones_like(voxel_indices_inverse))

    # Compute the mean of each voxel and fill the output tensor
    voxel_grid_downsampled = voxel_sum / voxel_count.unsqueeze(1)
    return voxel_grid_downsampled


def remove_statistical_outliers(
    voxel_grid: torch.Tensor, num_points: int, std_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove outliers from a voxel grid using pytorch3d knn points."""
    [nn_dists], _, _ = knn_points(voxel_grid[None, ...], voxel_grid[None, ...], K=num_points)
    avg_dists = nn_dists.mean(dim=-1)
    avg_dist = avg_dists.mean()
    std_dist = avg_dists.std()

    # Calculate threshold
    dist_threshold = avg_dist + std_ratio * std_dist
    nn_mask = avg_dists < dist_threshold
    nn_voxel_grid = voxel_grid[nn_mask]
    return nn_voxel_grid, nn_mask


def density_threshold_mask(
    alpha: Float[torch.Tensor, "n_x n_y n_z"],
    threshold: float,
    min_bounds: Tuple[float, float, float],
    max_bounds: Tuple[float, float, float],
) -> Float[torch.Tensor, "n 3"]:
    """
    Apply a simple density threshold to keep voxels above the threshold.
    This keeps both surface and interior points.
    
    Args:
        alpha: Density values in a 3D grid
        threshold: Density threshold value
        min_bounds: Minimum bounds of the voxel grid
        max_bounds: Maximum bounds of the voxel grid
        
    Returns:
        Points corresponding to voxels above the threshold
    """
    # Create mask for voxels above threshold
    mask = alpha > threshold
    
    if not torch.any(mask):
        print(f"No voxels found above threshold {threshold}. Max alpha: {alpha.max().item()}")
        # Return a small number of points with highest alpha to avoid empty tensor
        if alpha.max() > 0:
            top_k = min(100, alpha.numel())
            flat_indices = torch.topk(alpha.flatten(), k=top_k)[1]
            indices = torch.stack(torch.where(alpha.flatten()[flat_indices[0]] == alpha)).squeeze()
        else:
            print("WARNING: All alpha values are zero!")
            return torch.zeros((0, 3), device=alpha.device)
    else:
        # Get indices of voxels above threshold
        indices = torch.stack(torch.where(mask)).transpose(0, 1)
    
    # Convert indices to world coordinates
    grid_shape = torch.tensor(alpha.shape, device=alpha.device)
    min_bounds_tensor = torch.tensor(min_bounds, device=alpha.device)
    max_bounds_tensor = torch.tensor(max_bounds, device=alpha.device)
    
    # Scale indices to world coordinates
    voxel_size = (max_bounds_tensor - min_bounds_tensor) / grid_shape
    points = indices.float() * voxel_size + min_bounds_tensor
    
    print(f"Density threshold found {len(points)} voxels above threshold {threshold}")
    return points
