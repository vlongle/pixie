import sys
import torch
import numpy as np
from typing import Tuple, Dict, List
import trimesh
from f3rm.features.clip import clip
from f3rm.features.clip_extract import CLIPArgs
from pixie.utils import str2bool, load_json
import logging
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import trimesh
import json
import os
from pixie.utils import set_logger

def get_initial_voxel_grid_from_saved(
    grid_feature_path: str,
    occupancy_path: str = None,  # Deprecated: kept for RGB color mapping only
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Get the feature grid from a saved file masked by the occupancy grid provided
    by the occupancy_path. `occupancy_path` was precomputed using alpha thresholding +
    removing `black` background + connected component outlier removal using DBSCAN. See 
    `voxelize.py` for more details.
    
    Args:
        grid_feature_path: Path to the saved feature grid metadata (.npz file)
        query: Text query to filter voxels
        clip_model: CLIP model for text encoding
        device: Device to use for computation
        alpha_threshold: Threshold for density values
        softmax_temperature: Temperature for softmax when computing language probabilities
        voxel_size: Size of each voxel
        point_cloud_path: Path to pre-filtered point cloud (if None, will use default path)
        
    Returns:
        Tuple containing:
        - voxel_grid: Tensor of shape (num_voxels, 3) containing filtered voxel coordinates
        - voxel_sims: Tensor of shape (num_voxels) containing similarities with language query
        - metrics: Dictionary with metrics about filtering process
    """
    logging.info(f"Loading feature grid from {grid_feature_path}...")
    
    # Load metadata
    metadata = np.load(grid_feature_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    grid_shape = metadata['grid_shape']
    
    logging.info(f"Bounds: min={min_bounds}, max={max_bounds}")
    
    # Load features
    features_path = grid_feature_path.replace('.npz', '_features.npy')
    logging.info(f"Loading features from {features_path}...")
    features = np.load(features_path)
    
    # Track metrics
    metrics = {"initial": np.prod(grid_shape)}
    
    # Load occupancy mask to drive voxel selection (authoritative mask)
    mask_path = grid_feature_path.replace('.npz', '_mask.npy')
    assert os.path.exists(mask_path), f"Mask not found at {mask_path}. Please run voxelization first."
    mask_np = np.load(mask_path).astype(bool)

    # Create coordinate grid and select masked voxels (ordering matches mask flatten in C-order)
    logging.info("Creating coordinate grid from metadata and applying occupancy mask...")
    x = torch.linspace(min_bounds[0], max_bounds[0], grid_shape[0], device=device)
    y = torch.linspace(min_bounds[1], max_bounds[1], grid_shape[1], device=device)
    z = torch.linspace(min_bounds[2], max_bounds[2], grid_shape[2], device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D,H,W,3)

    # Flatten features in C-order and select exactly masked indices
    features_tensor = torch.from_numpy(features).to(device)
    features_flat = features_tensor.reshape(-1, features_tensor.shape[-1])
    linear_mask = torch.from_numpy(mask_np.reshape(-1)).to(device)
    features_filtered = features_flat[linear_mask]

    # Keep coords for downstream RGB mapping and saving utilities
    coords_filtered = coords[torch.from_numpy(mask_np)].to(device)

    metrics["masked_voxels"] = int(linear_mask.sum().item())
    logging.info(f"Selected {metrics['masked_voxels']} voxels from mask")

    return features_filtered, coords_filtered, metrics


def load_occupancy_grid(occupancy_path: str, device: str = "cuda"):
    pc = trimesh.load(occupancy_path)
    points = np.asarray(pc.vertices)
    return torch.tensor(points, dtype=torch.float32, device=device)



def run_clip(queries, features_filtered, softmax_temperature, device="cuda"):
    # Load CLIP model
    clip_model, _ = clip.load(CLIPArgs.model_name, device=device)
    
    # Normalize features
    features_filtered = features_filtered.to(torch.float32)
    features_filtered /= features_filtered.norm(dim=-1, keepdim=True)
    
    # Encode all part queries
    with torch.no_grad():
        text_inputs = clip.tokenize(queries).to(device)
        query_embs = clip_model.encode_text(text_inputs).float()
        query_embs /= query_embs.norm(dim=-1, keepdim=True)
    
    # Compute similarities between each voxel and each part query
    # Shape: (num_voxels, num_parts)
    similarities = features_filtered @ query_embs.T


    scaled_similarities = similarities / softmax_temperature
    
    # Convert scaled similarities to probabilities via softmax.
    probabilities = torch.nn.functional.softmax(scaled_similarities, dim=1)

    return probabilities, similarities


def clip_part_segmentation(
   grid_feature_path: str,
   part_queries: List[str],
   occupancy_path: str = None,
   device: str = "cuda",
    softmax_temperature: float = 0.1,  # Added temperature parameter for sharpening

):
    """
    Perform part-based segmentation of a voxel grid using CLIP.
    
    Assign each voxel to one of the part queries.
    
    Args:
        grid_feature_path: Path to the saved feature grid metadata (.npz file)
        part_queries: List of text queries representing different parts
        device: Device to use for computation
        occupancy_path: Path to pre-filtered point cloud
        
    Returns:
        Tuple containing:
        - coords_filtered: Tensor of shape (num_voxels, 3) containing voxel coordinates
        - part_labels: Tensor of shape (num_voxels) containing part indices (0 to len(part_queries)-1)
        - part_scores: Tensor of shape (num_voxels) containing similarity scores for the assigned parts
        - metrics: Dictionary with metrics about the segmentation process
    """
    features_filtered, coords_filtered, metrics = get_initial_voxel_grid_from_saved(
        grid_feature_path,
        device=device,
        occupancy_path=occupancy_path,
    )
    

    
    logging.info(f"features_filtered: {features_filtered.shape}")
    probabilities, _ = run_clip(part_queries, features_filtered, softmax_temperature, device=device)

    # Get the index of the part with highest similarity for each voxel
    # Shape: (num_voxels)
    part_labels = torch.argmax(probabilities, dim=1)
    
    # Get the similarity score for the assigned part
    # Shape: (num_voxels)
    part_scores = torch.gather(probabilities, 1, part_labels.unsqueeze(1)).squeeze(1)
    
    
    # Get the coordinates for each voxel (assuming they're available from the first function)
    # This needs to be fixed as coords_filtered isn't returned by get_initial_voxel_grid_from_saved
    # For now, we'll need to reconstruct the coordinates
    
    metrics["num_parts"] = len(part_queries)
    
    # Count voxels assigned to each part
    for i, query in enumerate(part_queries):
        part_count = (part_labels == i).sum().item()
        metrics[f"part_{i}_{query}"] = part_count
        logging.info(f"Part {i} ({query}): {part_count} voxels")
    
    return coords_filtered, part_labels, part_scores, metrics

import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import mode
from tqdm import tqdm

def local_post_process_segmentation(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    k: int = 200,
) -> torch.Tensor:
    """
    Perform local post-processing on segmentation results using k-nearest neighbors majority voting.
    
    Args:
        coords: Tensor of shape (num_points, 3) containing point coordinates.
        part_labels: Tensor of shape (num_points) containing segmentation labels.
        k: Number of nearest neighbors to consider for voting.
        
    Returns:
        new_labels: Tensor of shape (num_points) with updated labels after local post-processing.
    """

    # Convert tensors to NumPy arrays
    coords_np = coords.cpu().numpy()
    labels_np = part_labels.cpu().numpy()
    
    # Build a KDTree for fast neighbor search
    tree = KDTree(coords_np)
    new_labels_np = labels_np.copy()
    
    logging.info(">>>> LOCAL POST-PROCESSING")
    # For each point, query the k nearest neighbors and take a majority vote.
    for i, point in tqdm(enumerate(coords_np), total=len(coords_np), desc="Local Post-Processing"):
        # Query the k nearest neighbors (including the point itself)
        _, indices = tree.query(point.reshape(1, -1), k=k)
        neighbor_labels = labels_np[indices[0]]
        # Compute the mode (most frequent label) among the neighbors
        m = mode(neighbor_labels, keepdims=False)
        new_labels_np[i] = m.mode
    
    # Return as a torch tensor on the original device
    return torch.tensor(new_labels_np, device=part_labels.device)




def save_segmented_point_cloud(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    output_dir: str,
    cmap_name: str = 'tab10',
    original_pc_path: str = None,
    part_queries: List[str] = None,
    material_props: Dict[str, Dict[str, float]] = None,
    grid_feature_path: str = None,  # Added parameter for the original grid path
    background_id: int = 7  # Added parameter for background material ID
):
    """
    Save segmented point cloud to a PLY file with colors based on part labels.
    
    Args:
        coords: Tensor of shape (num_points, 3) containing point coordinates
        part_labels: Tensor of shape (num_points) containing part indices
        output_dir: Directory to save the output files
        cmap_name: Name of the colormap to use for part colors
        original_pc_path: Path to the original point cloud file (required if use_actual_rgb=True)
        part_queries: List of part query strings corresponding to part_labels
        material_dict_path: Path to JSON file mapping part queries to material properties
        grid_feature_path: Path to the original feature grid metadata (.npz file)
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths within the directory
    rgb_output_path = os.path.join(output_dir, "segmented_rgb.ply")
    semantic_output_path = os.path.join(output_dir, "segmented_semantics.ply")
    material_grid_path = os.path.join(output_dir, "material_grid.npy")
    
    # Convert tensors to numpy arrays
    coords_np = coords.cpu().numpy()
    part_labels_np = part_labels.cpu().numpy()
    
    assert len(part_labels_np) == len(coords_np), f"part_labels_np and coords_np must have the same length. len(part_labels_np): {len(part_labels_np)}, len(coords_np): {len(coords_np)}. Mismatch is likely due to new voxelization and cached part_labels_np. Try re-running with overwrite=True to recompute part_labels"
    
    # Initialize colors array for RGB and semantic colors
    rgb_colors = np.zeros((coords_np.shape[0], 4), dtype=np.float32)
    semantic_colors = np.zeros((coords_np.shape[0], 4), dtype=np.float32)
    
    # Initialize material property arrays
    density = np.zeros(coords_np.shape[0], dtype=np.float32)
    E = np.zeros(coords_np.shape[0], dtype=np.float32)
    nu = np.zeros(coords_np.shape[0], dtype=np.float32)
    material_id = np.zeros(coords_np.shape[0], dtype=np.int32)
    

    
    # Get RGB colors from original point cloud if available
    if original_pc_path:
        logging.info(">>> LOADING ORIGINAL RGB")
        # Load original point cloud to get RGB values
        original_pc = trimesh.load(original_pc_path)
        original_vertices = np.asarray(original_pc.vertices)
        original_colors = np.asarray(original_pc.colors)
        
        # Normalize colors if needed
        if original_colors.max() > 1.0:
            original_colors = original_colors / 255.0
            
        # We need to map the filtered coordinates back to the original point cloud
        # This is a simple implementation that finds the nearest neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(original_vertices)
        _, indices = tree.query(coords_np, k=1)
        
        # Get the corresponding colors
        rgb_colors[:, :3] = original_colors[indices, :3]
        rgb_colors[:, 3] = 1.0  # Full alpha
    else:
        # If no original point cloud, use white for RGB
        rgb_colors[:, :3] = 1.0  # White
        rgb_colors[:, 3] = 1.0  # Full alpha
    
    # Create semantic colors based on part labels
    logging.info(">>> CREATING SEMANTIC COLORS")
    # Create a colormap with distinct colors for each part
    num_parts = part_labels_np.max() + 1
    cmap = plt.colormaps[cmap_name]
    
    # Generate colors for each point based on its part label
    for i in range(num_parts):
        mask = (part_labels_np == i)
        if not np.any(mask):
            continue
            
        base_color = cmap(i % cmap.N)  # RGBA tuple
        semantic_colors[mask] = np.array(base_color)
    
    # Assign material properties based on part labels
    for i in range(part_labels_np.max() + 1):
        mask = (part_labels_np == i)
        if not np.any(mask):
            continue
        
        # Get part query string for this label
        part_name = part_queries[i]
        
        assert part_name in material_props, f"part_name `{part_name}` not found in material_props. Material props: {material_props}"
        props = material_props[part_name]
        density[mask] = props.get("density", 200)
        E[mask] = props.get("E", 2e6)
        nu[mask] = props.get("nu", 0.4)
        material_id[mask] = props.get("material_id", 0)
        logging.info(f"Applied material properties for {part_name}: {props}")
    
    # Save both RGB and semantic point clouds
    
    # 1. Save RGB point cloud
    # Convert floating point colors [0,1] to uint8 [0,255]
    rgb_colors_uint8 = (rgb_colors * 255).astype(np.uint8)
    
    # Create structured array for RGB PLY file
    vertex_data_rgb = np.zeros(
        coords_np.shape[0],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('part_label', 'i4'), ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
            ('material_id', 'i4')
        ]
    )
    
    # Fill in the data for RGB point cloud
    vertex_data_rgb['x'] = coords_np[:, 0]
    vertex_data_rgb['y'] = coords_np[:, 1]
    vertex_data_rgb['z'] = coords_np[:, 2]
    vertex_data_rgb['red'] = rgb_colors_uint8[:, 0]
    vertex_data_rgb['green'] = rgb_colors_uint8[:, 1]
    vertex_data_rgb['blue'] = rgb_colors_uint8[:, 2]
    vertex_data_rgb['alpha'] = rgb_colors_uint8[:, 3]
    vertex_data_rgb['part_label'] = part_labels_np
    vertex_data_rgb['density'] = density
    vertex_data_rgb['E'] = E
    vertex_data_rgb['nu'] = nu
    vertex_data_rgb['material_id'] = material_id
    
    # Create PLY element and save RGB file
    vertex_element_rgb = PlyElement.describe(vertex_data_rgb, 'vertex')
    PlyData([vertex_element_rgb], text=False).write(rgb_output_path)
    logging.info(f"RGB point cloud saved to {rgb_output_path}")
    
    # 2. Save semantic point cloud
    # Convert semantic colors to uint8
    semantic_colors_uint8 = (semantic_colors * 255).astype(np.uint8)
    
    # Create structured array for semantic PLY file
    vertex_data_semantic = np.zeros(
        coords_np.shape[0],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('part_label', 'i4'), ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
            ('material_id', 'i4')
        ]
    )
    
    # Fill in the data for semantic point cloud
    vertex_data_semantic['x'] = coords_np[:, 0]
    vertex_data_semantic['y'] = coords_np[:, 1]
    vertex_data_semantic['z'] = coords_np[:, 2]
    vertex_data_semantic['red'] = semantic_colors_uint8[:, 0]
    vertex_data_semantic['green'] = semantic_colors_uint8[:, 1]
    vertex_data_semantic['blue'] = semantic_colors_uint8[:, 2]
    vertex_data_semantic['alpha'] = semantic_colors_uint8[:, 3]
    vertex_data_semantic['part_label'] = part_labels_np
    vertex_data_semantic['density'] = density
    vertex_data_semantic['E'] = E
    vertex_data_semantic['nu'] = nu
    vertex_data_semantic['material_id'] = material_id
    logging.info("[SEGMENTATION] Statistics:")
    logging.info(f"material_id: {np.unique(material_id, return_counts=True)}")
    logging.info(f"part_labels: {np.unique(part_labels_np, return_counts=True)}")
    logging.info(f"density: {np.mean(density)}")
    logging.info(f"E: {np.mean(E)}")
    logging.info(f"nu: {np.mean(nu)}")
    
    # Create PLY element and save semantic file
    vertex_element_semantic = PlyElement.describe(vertex_data_semantic, 'vertex')
    PlyData([vertex_element_semantic], text=False).write(semantic_output_path)
    logging.info(f"Semantic point cloud saved to {semantic_output_path}")
    
    # 3. Save material properties for the entire voxel grid
    if grid_feature_path is not None:
        logging.info(">>> CREATING MATERIAL GRID")
        # Load metadata from the original grid
        metadata = np.load(grid_feature_path)
        min_bounds = metadata['min_bounds']
        max_bounds = metadata['max_bounds']
        grid_shape = metadata['grid_shape']
        
        logging.info(f"Grid shape: {grid_shape}")
        
        # Create material property grids with the same shape as the original grid
        # Each grid will have 4 channels: density, E, nu, material_id
        material_grid = np.zeros((*grid_shape, 4), dtype=np.float32)
        
        # Set default values for background (material_id=background_id)
        material_grid[..., 0] = 0  # density = 0
        material_grid[..., 1] = 0  # E = 0
        material_grid[..., 2] = 0  # nu = 0
        material_grid[..., 3] = background_id  # material_id = background_id
        # material_grid = np.ones((*grid_shape, 4), dtype=np.float32) * -1 ## default to -1
        
        # Use the authoritative occupancy mask to place values exactly
        mask_path = grid_feature_path.replace('.npz', '_mask.npy')
        assert os.path.exists(mask_path), f"Mask not found: {mask_path}"
        mask = np.load(mask_path).astype(bool)

        flat_idx = np.flatnonzero(mask.ravel(order='C'))
        assert len(flat_idx) == len(coords_np), (
            f"Mask/coords length mismatch: mask has {len(flat_idx)} true voxels, "
            f"coords has {len(coords_np)} points. Ensure coords come from mask.")

        material_grid_flat = material_grid.reshape(-1, 4)
        material_grid_flat[flat_idx, 0] = density
        material_grid_flat[flat_idx, 1] = E
        material_grid_flat[flat_idx, 2] = nu
        material_grid_flat[flat_idx, 3] = material_id

        # logging.info(f"Number of points in point cloud: {len(coords_np)}")
        # logging.info(f"Number of unique voxels assigned: {len(unique_voxels_assigned)}")
        # logging.info(f"Difference (duplicate mappings): {len(coords_np) - len(unique_voxels_assigned)}")

        # material_id_count = (material_grid[:, :, :, 3] == 7).sum()
        # logging.info(f"AFTER ASSIGNMENT: Number of material_id == 7 in material_grid: {material_id_count}")
        # logging.info(f"Number of voxels with material_id != BACKGROUND_ID: {1000000 - material_id_count}")
        
        # Save the material grid
        np.save(material_grid_path, 
                 material_grid)
        logging.info(f"Material grid saved to {material_grid_path}")
        
        # Also save each property as a separate file for easier visualization
        np.save(os.path.join(output_dir, "density_grid.npy"), material_grid[..., 0])
        np.save(os.path.join(output_dir, "E_grid.npy"), material_grid[..., 1])
        np.save(os.path.join(output_dir, "nu_grid.npy"), material_grid[..., 2])
        np.save(os.path.join(output_dir, "material_id_grid.npy"), material_grid[..., 3])
        logging.info(f"Individual material property grids saved to {output_dir}")




import torch
import numpy as np
from scipy.spatial import KDTree
from collections import deque

def build_adjacency(coords, radius=0.05):
    """
    Build an adjacency list for each point i, containing indices of neighbors
    within the specified 'radius'.
    
    Args:
        coords (np.ndarray): Array of shape (N, 3) with point coordinates.
        radius (float): Neighborhood radius for adjacency.
        
    Returns:
        adjacency (list[list[int]]): adjacency[i] is a list of neighbor indices for point i.
    """
    n_points = coords.shape[0]
    tree = KDTree(coords)
    adjacency = [[] for _ in range(n_points)]
    
    # For each point, find all neighbors within 'radius'
    for i in tqdm(range(n_points), total=n_points):
        neighbor_indices = tree.query_ball_point(coords[i], r=radius)
        adjacency[i] = neighbor_indices
    
    return adjacency

def get_connected_components(adjacency, labels):
    """
    Identify connected components for each label using BFS.
    
    Args:
        adjacency (list[list[int]]): adjacency[i] is a list of neighbors for point i.
        labels (np.ndarray): shape (N,) of integer labels.
        
    Returns:
        label_to_components (dict): 
            { label_value: [ list_of_components_for_this_label, ... ], ... }
        Each component is a list of point indices.
    """
    visited = set()
    label_to_components = {}
    n_points = len(labels)
    
    for i in tqdm(range(n_points), total=n_points):
        if i not in visited:
            current_label = labels[i]
            # BFS to gather connected points with the same label
            queue = deque([i])
            component = []
            visited.add(i)
            
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in adjacency[node]:
                    if nb not in visited and labels[nb] == current_label:
                        visited.add(nb)
                        queue.append(nb)
            
            label_to_components.setdefault(current_label, []).append(component)
    
    return label_to_components

def reassign_small_components(adjacency, labels, label_to_components):
    """
    For each label, keep only the largest connected component. All other small components
    get relabeled to a special debug label (-1).

    Args:
        adjacency (list[list[int]]): adjacency[i] is list of neighbor indices.
            These neighbors are points within the radius specified in build_adjacency().
        labels (np.ndarray): current labels for each point.
        label_to_components (dict): { label_value: [components], ... }
    """
    for lbl, components in label_to_components.items():
        logging.info(f">>> LABEL: {lbl} NUM COMPONENTS: {len(components)}")
        if len(components) <= 1:
            # There's only one component or none for this label, nothing to fix
            continue
        
        # Sort components by size descending
        components.sort(key=len, reverse=True)
        largest_component = components[0]
        small_components  = components[1:]
        
        # Keep the largest connected component with the current label
        # For each smaller component, relabel those points to -1 (debug label)
        for comp in small_components:
            for idx in comp:
                # Instead of using majority vote among neighbors, 
                # simply assign a special debug label (-1)
                labels[idx] = -1

                # The original code below used majority voting among neighbors:
                # neighbor_labels = []
                # for nb in adjacency[idx]:
                #     if nb not in comp:  # only consider neighbors outside this small component
                #         neighbor_labels.append(labels[nb])
                # 
                # if len(neighbor_labels) == 0:
                #     # If isolated (rare), keep the original label or do a fallback
                #     continue
                # 
                # # Majority vote among neighbor labels
                # neighbor_labels = np.array(neighbor_labels)
                # new_label = _majority_vote(neighbor_labels)
                # labels[idx] = new_label

def _majority_vote(label_array):
    """Return the most frequent label in label_array."""
    vals, counts = np.unique(label_array, return_counts=True)
    return vals[np.argmax(counts)]

def connected_component_cleanup(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    radius: float = 0.05
) -> torch.Tensor:
    """
    High-level function that:
    1) Builds adjacency among points (within 'radius').
    2) Finds connected components for each label.
    3) Keeps only the largest connected component for each label, re-labels smaller 'islands.'
    
    Args:
        coords (torch.Tensor): shape (N, 3) of point coordinates.
        part_labels (torch.Tensor): shape (N,) of integer labels.
        radius (float): neighborhood radius for adjacency graph.
    
    Returns:
        updated_labels (torch.Tensor): shape (N,) updated segmentation labels.
    """
    device = part_labels.device
    
    # Convert to numpy
    coords_np = coords.cpu().numpy()
    labels_np = part_labels.cpu().numpy()
    
    # 1) Build adjacency graph
    adjacency = build_adjacency(coords_np, radius=radius)
    
    # 2) Get connected components for each label
    label_to_components = get_connected_components(adjacency, labels_np)
    
    # 3) Re-label smaller "islands" if a label must be unique
    reassign_small_components(adjacency, labels_np, label_to_components)
    
    # Convert back to torch
    updated_labels = torch.from_numpy(labels_np).to(device)
    return updated_labels



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_feature_path", type=str, required=True)
    parser.add_argument("--occupancy_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--part_queries", type=str, required=True)
    parser.add_argument("--material_dict_path", type=str, required=True,
                        help="Path to JSON file mapping part queries to material properties")
    parser.add_argument("--use_spatial_smoothing", type=str2bool, default=False)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--background_id", type=int, default=7,
                        help="Material ID for background voxels")
    # parser.add_argument("--overwrite", type=str2bool, default=True)
    args = parser.parse_args()
    set_logger()

    assert os.path.exists(args.material_dict_path), f"material_dict_path {args.material_dict_path} does not exist"

    material_props = load_json(args.material_dict_path)
    logging.info(f"Loaded material properties from {args.material_dict_path}")
    if "material_dict" in material_props:
        material_props = material_props["material_dict"]
    part_queries = list(material_props.keys())

    labels_output_path = os.path.join(args.output_dir, "part_labels.npy")
    if args.overwrite or not os.path.exists(labels_output_path):
        coords_filtered, part_labels, part_scores, metrics = clip_part_segmentation(args.grid_feature_path, 
                                                                                    part_queries,
                                                                                    args.occupancy_path)
        if args.use_spatial_smoothing:
            logging.info(">>>> USING SPATIAL SMOOTHING")
            part_labels = local_post_process_segmentation(coords_filtered, part_labels)
        # Save part labels as a numpy array
        np.save(labels_output_path, part_labels.cpu().numpy())
        logging.info(f"Part labels saved to {labels_output_path}")
    else:
        part_labels = torch.from_numpy(np.load(labels_output_path))
        coords_filtered = load_occupancy_grid(args.occupancy_path)
    
    # Save all outputs to the specified directory
    save_segmented_point_cloud(coords_filtered, part_labels, args.output_dir, 
                               original_pc_path=args.occupancy_path,
                               part_queries=part_queries, 
                               material_props=material_props,
                               grid_feature_path=args.grid_feature_path,
                               background_id=args.background_id)  # Pass background_id from config