import numpy as np
import torch
import json
import os
import warp as wp
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from utils.transformation_utils import *
from collections import Counter
from pprint import pprint
from mpm_solver_warp.mpm_solver_warp import get_material_name
from plyfile import PlyData, PlyElement

# Constants
DEFAULT_VALUES = {
    'density': 1000.0,
    'E': 5000.0,
    'nu': 0.3,
    'part_label': 0,
    'material_id': 'stationary'  # Will be converted to ID later
}

class MaterialProperties:
    """Encapsulates material properties and provides helper methods"""
    
    def __init__(self, part_labels, densities, E_values, nu_values, material_ids, conf_values):
        self.properties = {
            'part_labels': part_labels,
            'density': densities, 
            'E': E_values,
            'nu': nu_values,
            'material_id': material_ids,
            'conf': conf_values
        }
    
    def get_defaults(self, n_particles):
        """Get default values for all properties"""
        defaults = {}
        for key, values in self.properties.items():
            if key == 'material_id':
                default_val = get_material_name("stationary")
            elif key in ['part_labels']:
                default_val = DEFAULT_VALUES['part_label']
            else:
                default_val = np.mean(values) if len(values) > 0 else DEFAULT_VALUES.get(key, 0.0)
            
            defaults[key] = np.full(n_particles, default_val, dtype=values.dtype if hasattr(values, 'dtype') else np.float32)
        return defaults
    
    def assign_from_neighbors(self, particle_idx, neighbor_indices, distances, weighted=False):
        """Assign material properties from K nearest neighbors"""
        results = {}
        
        # Distance-based weights
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights)
        
        for prop_name, prop_values in self.properties.items():
            neighbor_values = prop_values[neighbor_indices]
            
            if prop_name in ['material_id', 'part_labels']:
                # Categorical properties - use mode (weighted or unweighted)
                if weighted:
                    unique_vals, inv_indices = np.unique(neighbor_values, return_inverse=True)
                    votes = np.bincount(inv_indices, weights=weights)
                    results[prop_name] = unique_vals[np.argmax(votes)]
                else:
                    results[prop_name] = Counter(neighbor_values).most_common(1)[0][0]
            else:
                # Continuous properties - use mean (weighted or unweighted)
                if weighted:
                    results[prop_name] = np.dot(weights, neighbor_values)
                else:
                    results[prop_name] = np.mean(neighbor_values)
        
        return results


def transform_to_original_coordinates(positions, scale_origin, original_mean_pos, rotation_matrices):
    """Helper function to transform positions back to original coordinate system"""
    return apply_inverse_rotations(
        undotransform2origin(positions, scale_origin, original_mean_pos),
        rotation_matrices,
    )


def scene_bounds(positions):
    return {
        "x": [positions[:, 0].min(), positions[:, 0].max()],
        "y": [positions[:, 1].min(), positions[:, 1].max()],
        "z": [positions[:, 2].min(), positions[:, 2].max()],
    }

def save_points_as_ply(positions, colors, output_path, point_type="points"):
    """
    Save points as PLY file for visualization with trimesh or other tools.
    
    Args:
        positions: (N, 3) numpy array of point positions
        colors: (N, 3) numpy array of RGB colors (0-255) or (N,) array of labels
        output_path: Path to save the PLY file
        point_type: Description of what these points represent
    """
    if len(positions) == 0:
        print(f"No {point_type} to save")
        return
        
    # Ensure positions is the right shape
    positions = np.asarray(positions)
    assert positions.shape[1] == 3, f"Positions must be (N, 3), got {positions.shape}"
    
    # Handle colors - if it's labels, convert to colors
    colors = np.asarray(colors)
    if colors.ndim == 1:
        # Convert labels to colors
        unique_labels = np.unique(colors)
        # Create a simple color palette
        palette = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
        colors = np.array([color_map[label] for label in colors])
    
    # Ensure colors are uint8
    colors = colors.astype(np.uint8)
    
    # Create structured array for PLY format
    vertex_data = np.array([
        (positions[i, 0], positions[i, 1], positions[i, 2], 
         colors[i, 0], colors[i, 1], colors[i, 2])
        for i in range(len(positions))
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([vertex_element]).write(output_path)
    print(f"Saved {len(positions)} {point_type} to {output_path}")


def save_dbscan_debug_data(positions, labels, output_dir="dbscan_debug"):
    """
    Save DBSCAN clustering results as separate PLY files for debugging.
    
    Args:
        positions: (N, 3) numpy array of point positions  
        labels: (N,) numpy array of cluster labels from DBSCAN
        output_dir: Directory to save debug files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save noise points (label == -1)
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        noise_positions = positions[noise_mask]
        noise_colors = np.full((len(noise_positions), 3), [255, 0, 0])  # Red for noise
        save_points_as_ply(
            noise_positions, 
            noise_colors, 
            os.path.join(output_dir, "noise_points.ply"),
            "noise points"
        )
    
    # Save each cluster separately
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]
    
    if len(valid_labels) > 0:
        # Create color palette for clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_labels)))[:, :3] * 255
        
        for i, cluster_id in enumerate(valid_labels):
            cluster_mask = (labels == cluster_id)
            cluster_positions = positions[cluster_mask]
            cluster_color = np.full((len(cluster_positions), 3), cluster_colors[i % len(cluster_colors)])
            
            
            save_points_as_ply(
                cluster_positions,
                cluster_color,
                os.path.join(output_dir, f"cluster_{cluster_id}.ply"),
                f"cluster {cluster_id}"
            )
    
    # Save all points with cluster colors
    all_colors = np.zeros((len(positions), 3))
    for i, cluster_id in enumerate(valid_labels):
        cluster_mask = (labels == cluster_id)
        all_colors[cluster_mask] = cluster_colors[i % len(cluster_colors)]
    
    # Color noise points red
    noise_mask = (labels == -1)
    all_colors[noise_mask] = [255, 0, 0]
    
    save_points_as_ply(
        positions,
        all_colors,
        os.path.join(output_dir, "all_clusters.ply"),
        "all clustered points"
    )
    
    print(f"Saved DBSCAN debug data to {output_dir}/")
    print(f"  - {np.sum(noise_mask)} noise points")
    print(f"  - {len(valid_labels)} clusters")
    for i, cluster_id in enumerate(valid_labels):
        cluster_size = np.sum(labels == cluster_id)
        print(f"    Cluster {cluster_id}: {cluster_size} points")


def extract_material_properties(params):
    """Extract and convert material properties from params dictionary"""
    def to_numpy(tensor_or_array):
        return tensor_or_array.cpu().numpy() if torch.is_tensor(tensor_or_array) else tensor_or_array
    
    part_labels = to_numpy(params['part_labels'])
    densities = to_numpy(params['density']) 
    E_values = to_numpy(params['E'])
    nu_values = to_numpy(params['nu'])
    material_ids = to_numpy(params['material_id'])
    conf_values = params['conf']  # numpy array already
    
    return MaterialProperties(part_labels, densities, E_values, nu_values, material_ids, conf_values)


def perform_knn_smoothing(mpm_solver, params, device, scale_origin, original_mean_pos, 
                         rotation_matrices, k_smoothing_neighbors, nn_distance_threshold, 
                         weighted_assignment, debug):
    """Perform K-NN smoothing interpolation for material assignment"""
    n_particles = mpm_solver.n_particles
    material_props = extract_material_properties(params)
    
    # Check if smoothing is needed
    if len(material_props.properties['part_labels']) == n_particles:
        print(f">> Material field data matches MPM solver ({n_particles} particles). No K-NN smoothing applied.")
        return tuple(material_props.properties.values())
    
    print(f"Material field data ({len(material_props.properties['part_labels'])} particles) doesn't match MPM solver ({n_particles} particles). Performing K-NN smoothing interpolation (K={k_smoothing_neighbors}).")
    
    # Get and transform positions
    material_positions = params['pos'].cpu().numpy() if torch.is_tensor(params['pos']) else params['pos']
    mpm_positions_torch = mpm_solver.export_particle_x_to_torch().to(device)
    mpm_positions_transformed_np = transform_to_original_coordinates(
        undoshift2center111(mpm_positions_torch), 
        scale_origin, original_mean_pos, rotation_matrices
    ).detach().cpu().numpy()

    print("Material Field Point Cloud Bounds:", scene_bounds(material_positions))
    print("MPM Particles Bounds:", scene_bounds(mpm_positions_transformed_np))
    
    # Build KNN model and find neighbors
    nn_model = NearestNeighbors(n_neighbors=k_smoothing_neighbors, algorithm='auto').fit(material_positions)
    distances_all_k, k_indices = nn_model.kneighbors(mpm_positions_transformed_np)
    
    # Filter out particles that are too far
    too_far_mask = distances_all_k[:, 0] > nn_distance_threshold
    n_too_far, n_assigned = np.sum(too_far_mask), np.sum(~too_far_mask)
    print(f"Particles too far from nearest neighbor: {n_too_far}, Assigned: {n_assigned}")

    if debug and n_too_far > 0:
        save_points_as_ply(
            mpm_positions_transformed_np[too_far_mask],
            np.full((n_too_far, 3), [0, 0, 0]),
            "material_field_debug/too_far_particles.ply",
            "particles too far from material field"
        )

    assert n_too_far <= 0.1 * n_particles, f"[CRITICAL] More than 10% of particles are too far from nearest neighbor. Distance threshold: {nn_distance_threshold}."

    # Initialize arrays with defaults
    mapped_properties = material_props.get_defaults(n_particles)
    
    # Assign properties for particles that are close enough
    active_indices = np.where(~too_far_mask)[0]
    print(f"Smoothing with k_smoothing_neighbors: {k_smoothing_neighbors}")
    
    for i in tqdm(active_indices, desc="Smoothing material assignment", leave=False):
        neighbor_indices = k_indices[i]
        distances = distances_all_k[i]
        
        # Get assignments for this particle from its neighbors
        assignments = material_props.assign_from_neighbors(i, neighbor_indices, distances, weighted_assignment)
        
        # Update the mapped arrays
        for prop_name, value in assignments.items():
            mapped_properties[prop_name][i] = value
    
    print(f"K-NN smoothing complete. Avg distance to closest point: {distances_all_k[:, 0].mean():.6f}")
    print(f"Particles assigned default properties: {n_too_far}")
    
    return tuple(mapped_properties.values())


def apply_material_field_to_simulation(mpm_solver, params, device="cuda:0",
                                       scale_origin=None, original_mean_pos=None, rotation_matrices=None,
                                       only_handle_largest_cluster=True, fix_ground=True, 
                                       ground_delta_z=0.05, ground_buffer_xy=0.5,
                                       k_smoothing_neighbors=10,
                                       nn_distance_threshold=0.1,
                                       weighted_assignment=False,
                                       debug=False,
                                       ):
    """Apply material properties to particles based on material field data loaded from a point cloud."""
    # Check if material properties exist in the params
    missing = [k for k in ['part_labels', 'density', 'E', 'nu', 'material_id', 'conf'] if k not in params]
    assert not missing, f"Missing required keys: {missing}, Available: {list(params.keys())}"
    
    # Get the number of particles
    n_particles = mpm_solver.n_particles
    conf_values = params['conf']
    print(f"Loaded confidence values: shape={conf_values.shape}, range=[{conf_values.min():.3f}, {conf_values.max():.3f}]")
    
    # Perform K-NN smoothing if needed
    part_labels, densities, E_values, nu_values, material_ids, conf_values = perform_knn_smoothing(
        mpm_solver, params, device, scale_origin, original_mean_pos, rotation_matrices,
        k_smoothing_neighbors, nn_distance_threshold, weighted_assignment, debug
    )
    
    # Setup boundary conditions
    positions = mpm_solver.mpm_state.particle_x.numpy()
    bc_conditions = []
    
    if fix_ground:
        print("Adding ground boundary condition...")
        bc_conditions += fix_to_ground(mpm_solver, positions, ground_delta_z, ground_buffer_xy)
    
    bc_conditions += handle_stationary_clusters(
        mpm_solver, positions, material_ids, eps=0.03, min_samples=8, 
        start_time=0.0, end_time=1e9, buffer=0.1,
        only_handle_largest_cluster=only_handle_largest_cluster,
        debug_output_dir="stationary_clusters_debug", debug=debug
    )

    # Apply material properties to all particles
    _apply_material_properties_to_solver(mpm_solver, positions, densities, E_values, nu_values, material_ids, device)
    
    print("Material IDs: ", np.unique(material_ids, return_counts=True))
    return conf_values, bc_conditions


def _apply_material_properties_to_solver(mpm_solver, positions, densities, E_values, nu_values, material_ids, device):
    """Apply material properties to the MPM solver efficiently"""
    n_particles = len(positions)
    
    # Create material parameters list efficiently
    additional_params_list = [
        {
            "point": positions[i].tolist(),
            "size": [0.001, 0.001, 0.001],  # Tiny region for each particle
            "density": float(densities[i]),
            "E": float(E_values[i]),
            "nu": float(nu_values[i]),
            "material": int(material_ids[i]),
        }
        for i in tqdm(range(n_particles), desc="Applying material properties", leave=False)
    ]
    
    # Apply to solver
    material_params = {"additional_material_params": additional_params_list}
    mpm_solver.set_parameters_dict(material_params, device=device)
    mpm_solver.finalize_mu_lam(device=device)

def handle_stationary_clusters(mpm_solver, positions, material_ids,
                               eps=0.03, min_samples=10,
                               start_time=0.0, end_time=1e6, buffer=0.0,
                               only_handle_largest_cluster=True,
                               debug_output_dir="stationary_clusters_debug",
                               debug=False):
    """
    Automatically clusters stationary particles and creates one cuboid BC per cluster.
    
    Args:
        mpm_solver: your MPM_Simulator_WARP or similar solver instance
        positions: (N, 3) numpy array of particle positions
        material_ids: length-N array of material IDs for each particle
        eps: DBSCAN max distance for two samples to be in the same neighborhood
        min_samples: DBSCAN min number of samples to form a dense region
        start_time: BC start time
        end_time: BC end time
        buffer: an optional buffer to extend each bounding box in all directions (in world units)
        only_handle_largest_cluster: if True, only create a boundary condition for the largest cluster
        debug_output_dir: Directory to save debug PLY files
    """
    # 1) Filter only the stationary particles
    stationary_mask = (material_ids == get_material_name("stationary"))
    print(">>> stationary_mask: ", stationary_mask, "Number of stationary particles: ", np.sum(stationary_mask),
          "material_ids: ", np.unique(material_ids, return_counts=True))
    stationary_positions = positions[stationary_mask]
    print("[STATIONARY BC] Number of stationary points: ", len(stationary_positions))
    
    # Save all stationary particles for debugging
    if len(stationary_positions) > 0:
        stationary_colors = np.full((len(stationary_positions), 3), [0, 255, 0])  # Green for stationary
        save_points_as_ply(
            stationary_positions,
            stationary_colors,
            os.path.join(debug_output_dir, "stationary_particles.ply"),
            "stationary particles"
        )
    
    if len(stationary_positions) == 0:
        print("No stationary particles found; skipping cluster-based cuboid BC creation.")
        return []

    # 2) Run DBSCAN to find clusters among stationary positions
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(stationary_positions)
    
    # Save DBSCAN results for debugging
    if debug and len(stationary_positions) > 0:
        save_dbscan_debug_data(stationary_positions, labels, debug_output_dir)

    unique_labels = np.unique(labels)
    if len(unique_labels) == 1 and unique_labels[0] == -1:
        print("All stationary points marked as noise by DBSCAN; no cuboid BCs created.")
        return []
    
    print("[STATIONARY BC] Number of 'NOISE' labels: ", np.sum(labels == -1))
    # Filter out noise label (-1)
    valid_labels = unique_labels[unique_labels != -1]
    cluster_sizes = {label: np.sum(labels == label) for label in valid_labels}
    print("[STATIONARY BC] stationary cluster_sizes: ", cluster_sizes)
    # If only handling the largest cluster, find it and only process that one
    if only_handle_largest_cluster and len(valid_labels) > 1:
        # Count number of points in each cluster
        # Find the label of the largest cluster
        largest_cluster_label = max(cluster_sizes.items(), key=lambda x: x[1])[0]
        print(f"Only handling largest cluster (ID: {largest_cluster_label}) with {cluster_sizes[largest_cluster_label]} points")
        # Override valid_labels to only include the largest cluster
        valid_labels = np.array([largest_cluster_label])

    bc_conditions = []
    # 3) For each cluster, compute bounding box and add one cuboid BC
    for cluster_id in valid_labels:
        cluster_points = stationary_positions[labels == cluster_id]
        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)

        print(">> MIN_XYZ: ", min_xyz, "max_xyz: ", max_xyz)

        # bounding box center
        center = 0.5 * (min_xyz + max_xyz)
        # half-size
        halfsize = 0.5 * (max_xyz - min_xyz)

        # add optional buffer
        halfsize += buffer
        # 4) Create a single velocity-on-cuboid boundary condition for this cluster
        #    velocity=0, effectively pins that region for the entire simulation
        mpm_solver.set_velocity_on_cuboid(
            point=center.tolist(),
            size=halfsize.tolist(),
            velocity=[0.0, 0.0, 0.0],
            start_time=start_time,
            end_time=end_time,
            reset=1   # reset=1 forcibly sets velocity each step
        )
        print(">>> Created cuboid BC for cluster ", cluster_id, " at ", center.tolist(), " with size ", halfsize.tolist(),
        "this cluster has ", cluster_sizes[cluster_id], " points")
     
        # Collect the boundary condition data
        bc_conditions.append({
            "type": "stationary_cluster",
            "cluster_id": int(cluster_id),
            "point": center.tolist(),
            "size": halfsize.tolist(),
            "velocity": [0.0, 0.0, 0.0],
            "start_time": start_time,
            "end_time": end_time,
            "reset": 1,
            "cluster_size": int(cluster_sizes[cluster_id])
        })

    if only_handle_largest_cluster:
        print(f"Created cuboid BC for the largest stationary cluster.")
    else:
        print(f"Created cuboid BC for {len(valid_labels)} stationary cluster(s).")
    return bc_conditions


## NOTE: HACK: this assume a canonical pose where the object is upright with Z being the vertical direction
## this is not necessarily true for some real scenes. TODO: need to preprocess those scenes. Ignoring this issue for now.
def fix_to_ground(mpm_solver, positions, delta_z=0.02, buffer_xy=0.5, min_z_percentile=1,
                  start_time=0.0, end_time=1e6):
    """
    Creates a thin cuboid boundary condition at the base of the point cloud to fix it to an imaginary ground.
    
    Args:
        mpm_solver: your MPM_Simulator_WARP or similar solver instance
        positions: (N, 3) numpy array of particle positions
        delta_z: thickness of the ground boundary condition in z direction
        buffer_xy: additional buffer to extend the boundary in x and y directions (in world units)
        min_z_percentile: percentile to use for determining the lowest z position (1=min, 5=5th percentile)
        start_time: BC start time
        end_time: BC end time
        visualize: whether to visualize the ground boundary condition
        output_path: directory to save visualization if visualize=True
    """
    # Find the min and max positions in x, y dimensions
    min_xy = positions[:, :2].min(axis=0)
    max_xy = positions[:, :2].max(axis=0)
    
    # Calculate size of the ground plane in x,y dimensions
    size_xy = max_xy - min_xy
    
    # Find the lowest z-coordinate (using a percentile to avoid outliers)
    if min_z_percentile > 1:
        min_z = np.percentile(positions[:, 2], min_z_percentile)
    else:
        min_z = positions[:, 2].min()
    
    print(f"Ground BC - Position ranges: X:[{min_xy[0]:.4f}, {max_xy[0]:.4f}], Y:[{min_xy[1]:.4f}, {max_xy[1]:.4f}], Min Z:{min_z:.4f}")
    
    # Calculate center of the ground plane
    ground_center = [
        (min_xy[0] + max_xy[0]) / 2,  # x center
        (min_xy[1] + max_xy[1]) / 2,  # y center
        min_z + delta_z / 2  # z just above the lowest point
    ]
    
    # Calculate half-size of the ground plane (including buffer)
    ground_halfsize = [
        size_xy[0] / 2 + buffer_xy,  # x half-size with buffer
        size_xy[1] / 2 + buffer_xy,  # y half-size with buffer
        delta_z / 2  # z half-size (thin layer)
    ]
    
    # Create the velocity-on-cuboid boundary condition for the ground
    mpm_solver.set_velocity_on_cuboid(
        point=ground_center,
        size=ground_halfsize,
        velocity=[0.0, 0.0, 0.0],
        start_time=start_time,
        end_time=end_time,
        reset=1  # reset=1 forcibly sets velocity each step
    )
    
    print(f"Created ground boundary condition at {ground_center} with size {ground_halfsize}")
    ground_bc = {
        "type": "ground",
        "point": ground_center,
        "size": ground_halfsize,
        "velocity": [0.0, 0.0, 0.0],
        "start_time": start_time,
        "end_time": end_time,
        "reset": 1
    }
    return [ground_bc]
