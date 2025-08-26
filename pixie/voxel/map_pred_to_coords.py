import numpy as np
import argparse
import os
from plyfile import PlyData, PlyElement
import json
from pathlib import Path
import sys
import logging

# Add the parent directory to sys.path to import pixie utilities
sys.path.append(str(Path(__file__).parent.parent.parent))

from pixie.utils import resolve_paths, validate_config, load_json, set_logger
from pixie.training_utils import load_normalization_ranges
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose


def load_config(config_path="../../../config", config_name="config"):
    """
    Load and merge Hydra configuration.

    :param config_path: Path to the config directory
    :param config_name: Name of the main config file (without .yaml extension)
    :return: Merged configuration object
    """
    # Initialize Hydra
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)

    # Compose the configuration
    cfg = compose(config_name=config_name)
    
    # Load normalization ranges
    cfg = load_normalization_ranges(cfg)
    
    return cfg

def unscale_prediction(pred_tensor: np.ndarray, cfg: DictConfig):
    """Convert normalized network output back to original physical ranges but
    keep the tensor layout identical to the network output:
    (C,D,H,W) with C = 3 continuous + 8 categorical (=11).

    The first three channels are converted to real-world values, while the
    remaining categorical channels are left untouched.
    """
    # Number of continuous channels ------------------------------------------------
    CONT_CH = 3  # density, E, nu

    # Split (view only) channels ----------------------------------------------------
    cont = pred_tensor[:CONT_CH]             # (3,D,H,W)
    cat  = pred_tensor[CONT_CH:]             # (8,D,H,W) – untouched

    # The network's output is not strictly bounded, so we clip it to the expected [-1, 1] range.
    cont = np.clip(cont, -1.0, 1.0)

    # Allocate output array with the same shape as the input ------------------------
    out = pred_tensor.copy().astype(np.float32)  # (11,D,H,W)

    # Unscale continuous channels using the loaded normalization ranges ------------
    # Convert from [-1, 1] back to [min, max] range
    dens_log = (cont[0] + 1.0) * (cfg.training.density_max - cfg.training.density_min) / 2.0 + cfg.training.density_min
    out[0] = 10 ** dens_log
    logging.info(f"DENSITY: cont[0] {cont[0].mean()} mapped to {out[0].mean()}")

    E_log = (cont[1] + 1.0) * (cfg.training.E_max - cfg.training.E_min) / 2.0 + cfg.training.E_min
    out[1] = 10 ** E_log

    nu_scaled = cont[2]
    out[2] = (nu_scaled + 1.0) * (cfg.training.nu_max - cfg.training.nu_min) / 2.0 + cfg.training.nu_min

    # The categorical channels (indices 3:11) are already copied over via .copy()
    return out

def transform_nerf_to_world(ply_path: str, dataparser_path: str, world_output_path: str):
    """
    Transform coordinates from Nerf training space to world space using dataparser transforms.
    
    Args:
        ply_path: Path to input PLY file in Nerf training space
        dataparser_path: Path to dataparser_transforms.json file
        world_output_path: Path to save the transformed PLY file
    """

    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data

    ## dataparser contains the WORLD to NERF transform.
    # Load dataparser transforms
    logging.info(f"Loading dataparser transform from {dataparser_path}…")
    with open(dataparser_path, 'r') as f:
        dp_json = json.load(f)
    scale = float(dp_json["scale"])
    transform = np.asarray(dp_json["transform"], dtype=np.float32)  # shape (3,4)

    # Create transformation matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, :] = transform
    T_inv = np.linalg.inv(T)

    # Vectorised coordinate conversion
    coords_train = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T.astype(np.float32)
    coords_scaled = coords_train / scale
    coords_h = np.concatenate([coords_scaled, np.ones((coords_scaled.shape[0], 1), dtype=np.float32)], axis=1)
    coords_world = (T_inv @ coords_h.T).T[:, :3]

    # Create new vertex array for world frame
    vertex_world = vertex_data.copy()
    vertex_world['x'] = coords_world[:, 0]
    vertex_world['y'] = coords_world[:, 1]
    vertex_world['z'] = coords_world[:, 2]

    # Save transformed PLY
    vertex_world_element = PlyElement.describe(vertex_world, 'vertex')
    PlyData([vertex_world_element], text=False).write(world_output_path)
    logging.info(f"Saved WORLD-frame PLY to {world_output_path}")
    conf = vertex_world['conf']
    logging.info(f"conf {conf.shape} {conf.min()} {conf.max()} {conf.mean()}")

def get_mat_id(arr):
    if arr.shape == (1, 64, 64, 64):
        return arr[0]
    else: ## one hot
        return np.argmax(arr, axis=0)

def map_pred_to_ply(pred_path: str, mask_path: str, grid_feature_path: str, output_path: str,
                    obj_id: str, world_output_path: str = None, dataparser_path: str = None, cfg: DictConfig = None):
    """
    Map predictions from numpy arrays to a PLY file with material properties and part labels.
    
    Args:
        pred_path: Path to the prediction numpy array (shape: 11, 64, 64, 64)
        mask_path: Path to the mask numpy array (shape: 64, 64, 64)
        grid_feature_path: Path to the original voxel grid metadata (.npz file)
        output_path: Path to save the PLY file
        obj_id: Object ID
        world_output_path: Optional path to save the world frame PLY file
        dataparser_path: Optional path to dataparser_transforms.json
    """
    # Load metadata from the original grid
    logging.info(f"Loading metadata from {grid_feature_path}...")
    logging.info(f"path exists: {os.path.exists(grid_feature_path)}")

    metadata = np.load(grid_feature_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    grid_shape = metadata['grid_shape']
    
    logging.info(f"Grid shape: {grid_shape}")
    logging.info(f"Bounds: min={min_bounds}, max={max_bounds}")
    
    # Load predictions and mask
    logging.info(f"Loading predictions from {pred_path}...")
    scaled_pred = np.load(pred_path)

    # if scaled_pred.shape == (4, 64, 64, 64):
    #     print(f"Input prediction shape is {scaled_pred.shape}, converting to one-hot (11, 64, 64, 64)")
    #     cont_pred = scaled_pred[:3]
    #     material_ids = scaled_pred[3].astype(int)
    #     num_classes = 8  # From the expected shape (11 = 3 + 8)
    #     # Create one-hot encoding
    #     one_hot_seg = np.eye(num_classes, dtype=cont_pred.dtype)[material_ids]  # Shape (64, 64, 64, 8)
    #     one_hot_seg = np.transpose(one_hot_seg, (3, 0, 1, 2))  # Shape (8, 64, 64, 64)
    #     # Concatenate continuous predictions with one-hot segmentation
    #     scaled_pred = np.concatenate([cont_pred, one_hot_seg], axis=0)

    # assert scaled_pred.shape == (11, 64, 64, 64), f"scaled_pred.shape: {scaled_pred.shape}. Expected (11, 64, 64, 64)"
    logging.info(f"scaled Prediction shape: {scaled_pred.shape}")
    
    # Load config if not provided
    if cfg is None:
        cfg = load_config()
    
    pred = unscale_prediction(scaled_pred, cfg)
    logging.info(f"Unscaled Prediction shape: {pred.shape}")
    
    logging.info(f"Loading mask from {mask_path}...")
    mask = np.load(mask_path)
    logging.info(f"Mask shape: {mask.shape}")
    assert mask.shape == (64, 64, 64), f"mask.shape: {mask.shape}. Expected (64, 64, 64)"
    logging.info(f"Number of non-zero elements in mask: {np.sum(mask > 0)} out of {mask.size}")
    
    # Verify shapes
    pred_spatial_shape = pred.shape[1:4]  # Get spatial dimensions (64,64,64)
    if not np.array_equal(pred_spatial_shape, grid_shape):
        raise ValueError(f"Prediction spatial dimensions {pred_spatial_shape} do not match grid shape {grid_shape}")
    if not np.array_equal(mask.shape, grid_shape):
        raise ValueError(f"Mask shape {mask.shape} does not match grid shape {grid_shape}")
    
    # Split predictions into continuous and discrete parts
    cont = pred[:3, :]  # density, E, nu
    seg = pred[3:, :]  # material type probabilities
    
    # Get material_id from discrete predictions using argmax
    # material_id = np.argmax(seg, axis=0)
    material_id = get_mat_id(seg)
    
    # Create coordinate grid
    x = np.linspace(min_bounds[0], max_bounds[0], grid_shape[0])
    y = np.linspace(min_bounds[1], max_bounds[1], grid_shape[1])
    z = np.linspace(min_bounds[2], max_bounds[2], grid_shape[2])
    
    # Create meshgrid for coordinates
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # Stack coordinates
    coords = np.stack([grid_x, grid_y, grid_z], axis=-1)
    
    # Apply mask to get only valid points
    valid_mask = mask > 0
    valid_coords = coords[valid_mask]
    valid_density = cont[0][valid_mask]
    valid_E = cont[1][valid_mask]
    valid_nu = cont[2][valid_mask]
    valid_material_id = material_id[valid_mask]
    logging.info(f"All material_id {np.unique(material_id, return_counts=True)}")
    logging.info(f">> Valid material_id {np.unique(valid_material_id, return_counts=True)}")
    
    # Create structured array for PLY file
    vertex_data = np.zeros(
        len(valid_coords),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('part_label', 'i4'), ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
            ('material_id', 'i4'),
            ('conf', 'f4')
        ]
    )
    
    # Fill in the data
    vertex_data['x'] = valid_coords[:, 0]
    vertex_data['y'] = valid_coords[:, 1]
    vertex_data['z'] = valid_coords[:, 2]
    vertex_data['red'] = 255  # Default white color
    vertex_data['green'] = 255
    vertex_data['blue'] = 255
    vertex_data['alpha'] = 255
    vertex_data['part_label'] = valid_material_id
    vertex_data['density'] = valid_density
    vertex_data['E'] = valid_E
    vertex_data['nu'] = valid_nu
    vertex_data['material_id'] = valid_material_id
    # Confidence value: probability associated with the chosen material (max across categories)
    if seg.shape[0] > 1:  # One-hot encoded probabilities
        max_prob = np.max(seg, axis=0)
        valid_conf = max_prob[valid_mask]
    else:  # Class indices, so confidence is not available, default to 1.0
        valid_conf = np.ones_like(valid_material_id, dtype=np.float32)
    vertex_data['conf'] = valid_conf
    
    logging.info(f"valid_conf {valid_conf.shape}")
    logging.info(f"seg.shape {seg.shape}")
    logging.info(f"seg {seg.min()} {seg.max()} {seg.mean()}")

    logging.info("STATISTICS:")
    logging.info(f"Part_label {np.unique(vertex_data['part_label'], return_counts=True)}")
    logging.info(f"DENSITY: {vertex_data['density'].mean()} {vertex_data['density'].min()} {vertex_data['density'].max()}")
    logging.info(f"E: {vertex_data['E'].mean()} {vertex_data['E'].min()} {vertex_data['E'].max()}")
    logging.info(f"NU: {vertex_data['nu'].mean()} {vertex_data['nu'].min()} {vertex_data['nu'].max()}")
    
    # Create PLY element and save file
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=False).write(output_path)
    logging.info(f"Saved PLY file to {output_path} from {pred_path}")

    # ##############################
    # # NEW: Export to world frame #
    # ##############################
    logging.info(f"world_output_path {world_output_path}")
    if world_output_path is not None:
        if dataparser_path is None:
            # Heuristic: look for dataparser_transforms.json next to grid_feature_path
            dataparser_path = Path(grid_feature_path).parent / "dataparser_transforms.json"
            if not dataparser_path.exists():
                raise FileNotFoundError(
                    f"Could not find dataparser_transforms.json at {dataparser_path}. "
                    "Please provide the path using --dataparser_path argument."
                )
        
        transform_nerf_to_world(output_path, dataparser_path, world_output_path)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    set_logger()
    """Main function to map predictions to PLY file."""
    # Validate required mapping parameters
    if not cfg.mapping.pred_path:
        raise ValueError("mapping.pred_path is required")
    if not cfg.mapping.mask_path:
        raise ValueError("mapping.mask_path is required")
    if not cfg.mapping.grid_feature_path:
        raise ValueError("mapping.grid_feature_path is required")
    if not cfg.mapping.output_path:
        raise ValueError("mapping.output_path is required")
    if not cfg.mapping.obj_id:
        raise ValueError("mapping.obj_id is required")
    
    # Load normalization ranges
    cfg = load_normalization_ranges(cfg)
    
    map_pred_to_ply(
        cfg.mapping.pred_path, 
        cfg.mapping.mask_path, 
        cfg.mapping.grid_feature_path, 
        cfg.mapping.output_path, 
        cfg.mapping.obj_id, 
        world_output_path=cfg.mapping.world_output_path, 
        dataparser_path=cfg.mapping.dataparser_path, 
        cfg=cfg
    )

if __name__ == "__main__":
    main() 