import argparse
import logging
import os
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm

from pixie.utils import str2bool, set_logger
from f3rm_robot.load import load_nerfstudio_objaverse_outputs
from f3rm_robot.initial_proposals import dense_voxel_grid
from f3rm_robot.optimize import filter_gray_background, remove_floating_clusters, get_qp_feats


def extract_clip_voxel_grid(
    scene_path: str,
    output_path: str,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
    voxel_size: float = 0.01,
    batch_size: int = 4096,
    alpha_weighted: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    alpha_threshold_for_mask: float = 0.01,
    gray_threshold_for_mask: float = 0.05,
    run_outlier_filter: bool = True,
    nb_neighbors: int = 50,
    std_ratio: float = 4.0,
    min_cluster_pts: int = 10,
    eps_multiplier: float = 5.0,
) -> str:
    """
    Extract CLIP features in a voxel grid format from a trained F3RM model.
    
    Args:
        scene_path: Path to the trained F3RM model
        output_path: Path to save the extracted feature grid
        min_bounds: Minimum bounds of the voxel grid in world coordinates
        max_bounds: Maximum bounds of the voxel grid in world coordinates
        voxel_size: Size of each voxel
        batch_size: Number of voxels to process at once
        alpha_weighted: Whether to weight features by alpha (density)
        device: Device to use for computation
        alpha_threshold_for_mask: Threshold for occupancy mask generation
        gray_threshold_for_mask: Threshold for gray background filtering
        run_outlier_filter: Whether to run outlier filtering
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal
        min_cluster_pts: Minimum points for cluster removal
        eps_multiplier: Epsilon multiplier for DBSCAN clustering
        
    Returns:
        Path to the saved feature grid metadata
    """
    logging.info(f"Loading feature field from {scene_path}")
    load_state = load_nerfstudio_objaverse_outputs(scene_path)
    feature_field = load_state.feature_field_adapter()
    
    # Extract min/max bounds from the bounds tuple
    min_bounds = (bounds[0][0], bounds[1][0], bounds[2][0])
    max_bounds = (bounds[0][1], bounds[1][1], bounds[2][1])
    
    logging.info(f"Creating voxel grid with size {voxel_size}")
    voxel_grid = dense_voxel_grid(
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        voxel_size=voxel_size
    )
    original_shape = voxel_grid.shape[:-1]  # Exclude the last dimension (3 for xyz)
    logging.info(f"Voxel grid shape: {voxel_grid.shape}")
    
    # Flatten and move to device
    voxel_grid_flat = voxel_grid.reshape(-1, 3)
    voxel_grid_flat = torch.tensor(voxel_grid_flat, dtype=torch.float32, device=device)
    
    # Determine feature dimension
    with torch.no_grad():
        sample_output = feature_field(voxel_grid_flat[:1])
        feature_dim = sample_output["feature"].shape[-1]
    
    logging.info(f"Feature dimension: {feature_dim}")
    
    # Initialize arrays
    total_voxels = voxel_grid_flat.shape[0]
    features_cpu = np.zeros((total_voxels, feature_dim), dtype=np.float16)
    alphas_cpu = np.zeros((total_voxels, 1), dtype=np.float16)
    rgb_cpu = np.zeros((total_voxels, 3), dtype=np.float16)
    
    # Extract features in batches
    logging.info("Extracting features from voxel grid")
    with torch.no_grad():
        for i in tqdm(range(0, total_voxels, batch_size), desc="Extracting features"):
            end_idx = min(i + batch_size, total_voxels)
            batch = voxel_grid_flat[i:end_idx]
            
            # Get outputs from feature field
            outputs = feature_field(batch)
            alpha = feature_field.get_alpha(batch, voxel_size)
            
            # Get features - either raw or alpha-weighted
            if alpha_weighted:
                feature = get_qp_feats(outputs)
            else:
                feature = outputs["feature"]
            
            # Get RGB values
            rgb = feature_field.get_rgb(batch)
            
            # Move to CPU and convert to float16
            features_cpu[i:end_idx] = feature.cpu().to(torch.float16).numpy()
            alphas_cpu[i:end_idx] = alpha.cpu().to(torch.float16).numpy()
            rgb_cpu[i:end_idx] = rgb.cpu().to(torch.float16).numpy()
            
            # Free up GPU memory
            del outputs, alpha, feature, rgb
            torch.cuda.empty_cache()
    
    # Reshape to original grid shape
    logging.info("Reshaping arrays to grid format")
    features_reshaped = features_cpu.reshape(*original_shape, feature_dim)
    alphas_reshaped = alphas_cpu.reshape(*original_shape, 1)
    rgb_reshaped = rgb_cpu.reshape(*original_shape, 3)
    
    # Save data
    _save_voxel_data(
        output_path, features_reshaped, alphas_reshaped, rgb_reshaped,
        min_bounds, max_bounds, voxel_size, feature_dim, original_shape,
        alpha_weighted, alpha_threshold_for_mask
    )
    
    # Create occupancy mask
    _create_occupancy_mask(
        output_path, voxel_grid, alphas_reshaped, rgb_reshaped,
        alpha_threshold_for_mask, gray_threshold_for_mask,
        run_outlier_filter, nb_neighbors, std_ratio,
        min_cluster_pts, eps_multiplier, voxel_size, device
    )
    
    logging.info("Voxel grid extraction completed")
    return output_path


def _save_voxel_data(
    output_path: str,
    features: np.ndarray,
    alphas: np.ndarray,
    rgb: np.ndarray,
    min_bounds: Tuple[float, float, float],
    max_bounds: Tuple[float, float, float],
    voxel_size: float,
    feature_dim: int,
    grid_shape: Tuple[int, int, int],
    alpha_weighted: bool,
    alpha_threshold_for_mask: float,
) -> None:
    """Save voxel data to files."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metadata
    output_data = {
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "voxel_size": voxel_size,
        "feature_dim": feature_dim,
        "grid_shape": grid_shape,
        "alpha_weighted": alpha_weighted,
        "alpha_threshold_for_mask": alpha_threshold_for_mask,
    }
    np.savez_compressed(output_path, **output_data)
    
    # Save large arrays to separate files
    features_path = output_path.replace('.npz', '_features.npy')
    alphas_path = output_path.replace('.npz', '_alphas.npy')
    rgb_path = output_path.replace('.npz', '_rgb.npy')
    
    logging.info(f"Saving features to {features_path}")
    np.save(features_path, features)
    
    logging.info(f"Saving alphas to {alphas_path}")
    np.save(alphas_path, alphas)
    
    logging.info(f"Saving RGB to {rgb_path}")
    np.save(rgb_path, rgb)


def _create_occupancy_mask(
    output_path: str,
    voxel_grid: np.ndarray,
    alphas_reshaped: np.ndarray,
    rgb_reshaped: np.ndarray,
    alpha_threshold_for_mask: float,
    gray_threshold_for_mask: float,
    run_outlier_filter: bool,
    nb_neighbors: int,
    std_ratio: float,
    min_cluster_pts: int,
    eps_multiplier: float,
    voxel_size: float,
    device: str,
) -> None:
    """Create and save occupancy mask."""
    logging.info("Building occupancy mask (density + gray-BG + outlier/cluster filters)")
    
    # Apply density thresholding
    alphas_flat = torch.from_numpy(alphas_reshaped.reshape(-1)).to(device)
    density_mask = alphas_flat > alpha_threshold_for_mask
    idx_density = torch.where(density_mask)[0]
    
    coords_density = torch.tensor(
        voxel_grid.reshape(-1, 3)[idx_density.cpu().numpy()],
        dtype=torch.float32, device=device
    )
    
    rgb_density = torch.tensor(
        rgb_reshaped.reshape(-1, 3)[idx_density.cpu().numpy()],
        dtype=torch.float32, device=device
    )
    
    # Apply gray background filtering
    class _MockField:
        def get_rgb(self, _):
            return rgb_density
    
    non_bg_mask = filter_gray_background(
        coords_density, _MockField(),
        gray_threshold_for_mask, device, return_mask=True
    )
    idx_after_bg = idx_density[non_bg_mask]
    
    # Apply outlier and cluster filtering
    if run_outlier_filter and idx_after_bg.numel():
        pts_np = voxel_grid.reshape(-1, 3)[idx_after_bg.cpu().numpy()]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_np))
        
        # Statistical outlier removal
        pcd, ind_stat = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        idx_after_stat = idx_after_bg[ind_stat]
        
        # Floating cluster removal via DBSCAN
        labels = np.array(
            pcd.cluster_dbscan(
                eps=voxel_size * eps_multiplier,
                min_points=min_cluster_pts, print_progress=False
            )
        )
        keep_mask = labels != -1
        idx_final = idx_after_stat[keep_mask]
    else:
        idx_final = idx_after_bg
    
    # Create final mask
    final_mask_flat = torch.zeros_like(alphas_flat, dtype=torch.bool)
    final_mask_flat[idx_final] = True
    occupancy_mask = final_mask_flat.cpu().numpy().reshape(*alphas_reshaped.shape[:-1])
    
    # Save mask
    mask_path = output_path.replace('.npz', '_mask.npy')
    np.save(mask_path, occupancy_mask)
    logging.info(f"Saved occupancy mask to {mask_path}")


def compute_occupancy_point_cloud(
    feature_grid_path: str,
    alpha_threshold: float = 0.01,
    gray_threshold: float = 0.05,
    voxel_downsample_size: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    """
    Compute an occupancy point cloud from saved voxel grid features and alphas.
    
    Args:
        feature_grid_path: Path to the saved feature grid metadata (.npz file)
        alpha_threshold: Threshold for density values to consider a voxel occupied
        gray_threshold: Threshold for detecting gray background
        voxel_downsample_size: Size for downsampling the resulting point cloud
        device: Device to use for computation
        
    Returns:
        Tuple of (point_cloud, metrics_dict)
    """
    logging.info(f"Loading feature grid from {feature_grid_path}")
    metric = {}
    
    # Load metadata
    metadata = np.load(feature_grid_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    voxel_size = float(metadata['voxel_size'])
    grid_shape = metadata['grid_shape']
    
    metric['grid_shape'] = grid_shape
    metric["init_num_voxels"] = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    logging.info(f"Grid shape: {grid_shape}, voxel size: {voxel_size}")
    logging.info(f"Bounds: min={min_bounds}, max={max_bounds}")
    assert grid_shape[0] == grid_shape[1] == grid_shape[2] == 64, "Grid must be cubic == 64 (what we trained on)"
    
    # Load data
    alphas_path = feature_grid_path.replace('.npz', '_alphas.npy')
    rgb_path = feature_grid_path.replace('.npz', '_rgb.npy')
    
    logging.info(f"Loading alphas from {alphas_path}")
    alphas = np.load(alphas_path)
    
    logging.info(f"Loading RGB from {rgb_path}")
    rgb = np.load(rgb_path)
    
    # Convert to torch tensors
    alphas_tensor = torch.from_numpy(alphas).to(device)
    rgb_tensor = torch.from_numpy(rgb).to(device)
    
    # Create coordinate grid
    logging.info("Creating coordinate grid")
    x = torch.linspace(min_bounds[0], max_bounds[0], grid_shape[0], device=device)
    y = torch.linspace(min_bounds[1], max_bounds[1], grid_shape[1], device=device)
    z = torch.linspace(min_bounds[2], max_bounds[2], grid_shape[2], device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    # Flatten for processing
    coords_flat = coords.reshape(-1, 3)
    alphas_flat = alphas_tensor.reshape(-1, 1)
    rgb_flat = rgb_tensor.reshape(-1, 3)
    
    # Apply density thresholding
    logging.info(f"Applying density threshold {alpha_threshold}")
    density_mask = alphas_flat.squeeze(-1) > alpha_threshold
    coords_filtered = coords_flat[density_mask]
    rgb_filtered = rgb_flat[density_mask]
    metric["num_voxels_after_density_threshold"] = coords_filtered.shape[0]
    
    logging.info(f"After density filtering: {coords_filtered.shape[0]} points")
    
    # Apply gray background filtering
    class MockFeatureFieldAdapter:
        def get_rgb(self, points):
            return rgb_filtered
    
    mock_feature_field = MockFeatureFieldAdapter()
    
    logging.info(f"Applying gray background filtering with threshold {gray_threshold}")
    non_bg_mask = filter_gray_background(
        coords_filtered, mock_feature_field, gray_threshold, device, return_mask=True
    )
    
    # Handle different return types from filter_gray_background
    if isinstance(non_bg_mask, torch.Tensor) and non_bg_mask.shape == coords_filtered.shape:
        coords_filtered = non_bg_mask
        with torch.no_grad():
            rgb_filtered = mock_feature_field.get_rgb(coords_filtered)
    else:
        coords_filtered = coords_filtered[non_bg_mask]
        rgb_filtered = rgb_filtered[non_bg_mask]
    
    metric["num_voxels_after_gray_background_filtering"] = coords_filtered.shape[0]
    
    # Check if we have any points left
    if coords_filtered.shape[0] == 0:
        logging.warning("All voxels were filtered out. Creating empty point cloud.")
        logging.warning(f"This may indicate that your gray_threshold ({gray_threshold}) is too high.")
        logging.warning("Try lowering the gray_threshold (e.g., --gray_threshold 0.01) or adjusting alpha_threshold.")
        
        pcd = o3d.geometry.PointCloud()
        metric["num_voxels_after_downsampling"] = 0
        metric["num_voxels_after_outlier_removal"] = 0
        metric["num_voxels_after_floating_cluster_removal"] = 0
        metric["final_num_voxels"] = 0
        logging.info(f"Metrics: {metric}")
        return pcd, metric
    
    # Create point cloud
    logging.info("Creating point cloud")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_filtered.cpu().numpy())
    
    # Add colors (normalize if needed)
    if rgb_filtered.numel() > 0 and rgb_filtered.max() > 1.0:
        rgb_filtered = rgb_filtered / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb_filtered.cpu().numpy())
    
    # Downsample point cloud
    logging.info(f"Downsampling with voxel size {voxel_downsample_size}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_size)
    metric["num_voxels_after_downsampling"] = len(pcd.points)
    
    # Remove statistical outliers
    logging.info("Removing outliers")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    metric["num_voxels_after_outlier_removal"] = len(pcd.points)
    
    # Remove floating clusters
    logging.info("Removing floating clusters")
    pcd = remove_floating_clusters(pcd, min_points=10, eps=voxel_downsample_size*5)
    metric["num_voxels_after_floating_cluster_removal"] = len(pcd.points)
    
    logging.info(f"Final point cloud has {len(pcd.points)} points")
    metric["final_num_voxels"] = len(pcd.points)
    
    logging.info(f"Metrics: {metric}")
    return pcd, metric


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """
    Save a point cloud to a file.
    
    Args:
        pcd: Open3D point cloud
        output_path: Path to save the point cloud
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    logging.info(f"Saved point cloud to {output_path}")


def main() -> None:
    """Main function to run voxel extraction and point cloud generation."""
    parser = argparse.ArgumentParser(
        description="Extract CLIP features in voxel grid format from a trained F3RM model"
    )
    
    # Required arguments
    parser.add_argument("--scene", type=str, required=True, 
                       help="Path to the trained F3RM model")
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to save the extracted feature grid")
    
    # Bounds arguments
    parser.add_argument("--min_x", type=float, default=-0.5, help="Minimum x bound")
    parser.add_argument("--min_y", type=float, default=-0.5, help="Minimum y bound")
    parser.add_argument("--min_z", type=float, default=-0.5, help="Minimum z bound")
    parser.add_argument("--max_x", type=float, default=0.5, help="Maximum x bound")
    parser.add_argument("--max_y", type=float, default=0.5, help="Maximum y bound")
    parser.add_argument("--max_z", type=float, default=0.5, help="Maximum z bound")
    
    # Processing arguments
    parser.add_argument("--voxel_size", type=float, default=0.01, 
                       help="Size of each voxel")
    parser.add_argument("--batch_size", type=int, default=4096, 
                       help="Number of voxels to process at once")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    parser.add_argument("--alpha_weighted", type=str2bool, default=True, 
                       help="Weight features by alpha (density)")
    
    # Filtering arguments
    parser.add_argument("--alpha_threshold", type=float, default=0.01, 
                       help="Threshold for density values")
    parser.add_argument("--gray_threshold", type=float, default=0.05, 
                       help="Threshold for gray background filtering. Lower if all voxels are filtered out (e.g., 0.01).")
    
    # Output arguments
    parser.add_argument("--pc_output", type=str, 
                       help="Path to save the extracted point cloud")
    
    args = parser.parse_args()
    
    # Set up logging
    set_logger()
    
    # Prepare bounds
    bounds = ((args.min_x, args.max_x), (args.min_y, args.max_y), (args.min_z, args.max_z))
    
    # Extract voxel grid
    output_path = extract_clip_voxel_grid(
        args.scene, args.output, bounds,
        args.voxel_size, args.batch_size, args.alpha_weighted,
        args.device, args.alpha_threshold
    )
    
    # Generate point cloud
    pcd, metric = compute_occupancy_point_cloud(
        output_path, args.alpha_threshold, args.gray_threshold,
        args.voxel_size, args.device
    )
    
    # Save point cloud
    pc_output = args.pc_output or output_path.replace('.npz', '_pc.ply')
    save_point_cloud(pcd, pc_output)


if __name__ == "__main__":
    main()