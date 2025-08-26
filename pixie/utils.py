import os
import sys
import argparse
import logging
import json
import shutil
import objaverse
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import subprocess
from dotenv import load_dotenv
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import pickle
from typing import Tuple, Optional
import colorlog
import numpy as np

def set_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a colored logger using colorlog.
    
    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create colored handler
    handler = colorlog.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # If this is the root logger, also configure basicConfig
    if name is None:
        logging.basicConfig(level=level, handlers=[handler])
        
    return logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True','t', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    






def get_conda_env(env_name):
    conda_init = "source $(conda info --base)/etc/profile.d/conda.sh"
    acv = f"bash -c '{conda_init} && conda activate {env_name} &&"
    return acv


def _capture_command_output(cmd):
    """Run a command and capture its output."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    output_lines = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.rstrip())
            output_lines.append(output)
    
    return process.poll(), ''.join(output_lines)


def _is_blender_nerf_error_only(output_text):
    """Check if the output contains only the harmless BlenderNeRF camera error."""
    if "BlenderNeRF Camera" not in output_text or "not found" not in output_text:
        return False
    
    # Check for successful completion indicators
    success_indicators = ["Blender quit", "Finished", "Normalized scene exported"]
    return any(indicator in output_text for indicator in success_indicators)


def _is_gaussian_splatting_addon_error_only(output_text):
    """Check if the output contains only harmless Gaussian Splatting addon errors."""
    # Look for the specific addon unregister error
    gs_error_patterns = [
        "RuntimeError: unregister_class(...):, missing bl_rna attribute from '_RNAMeta' instance",
        "Exception in module unregister():",
        "gaussian_splatting_io"
    ]
    
    # Check if this is a GS addon error
    if not any(pattern in output_text for pattern in gs_error_patterns):
        return False
    
    # Check for successful completion indicators
    success_indicators = ["Blender quit", "Finished", "Video saved to", "✅"]
    return any(indicator in output_text for indicator in success_indicators)


def _check_for_errors(output_text):
    """Check output text for error patterns."""
    error_patterns = [
        "could not be opened: No such file or directory",
        "Error:",
        "ERROR:",
        "Failed to",
        "Traceback (most recent call last):",
    ]
    
    # Special handling for BlenderNeRF errors
    if _is_blender_nerf_error_only(output_text):
        logging.info("[RUN] Ignoring harmless BlenderNeRF camera error - command completed successfully")
        return False
    
    # Special handling for Gaussian Splatting addon errors
    if _is_gaussian_splatting_addon_error_only(output_text):
        logging.info("[RUN] Ignoring harmless Gaussian Splatting addon unregister error - command completed successfully")
        return False
    
    # Check for any error patterns
    for pattern in error_patterns:
        if pattern in output_text:
            return True
    
    return False

def run_cmd(cmd, step_name="", *, conda_env=None, allow_error=False, dry_run=False, check_output=False):
    """Execute a shell command with optional conda-environment wrapping.

    Parameters
    ----------
    cmd : str | list
        The command to run. Lists are automatically joined by spaces for
        convenience, so you can write ``["python", "script.py", "--arg", "val"]``
        instead of hand-crafting long f-strings.
    step_name : str, optional
        Human-readable label used in logs.
    conda_env : str, optional
        If given, the command is executed inside that conda environment using
        the same *conda activate* wrapper used elsewhere in the codebase.
    allow_error : bool, default False
        When *False*, a non-zero exit status will terminate the program.
    dry_run : bool, default False
        Log the command but do not execute it. Useful for debugging.
    check_output : bool, default False
        If True, captures output to check for errors (needed for Blender).
        This may break progress bars but is necessary for some commands.
    """

    # Convert list-based commands to a single string.
    if isinstance(cmd, list):
        cmd = " ".join(map(str, cmd))

    if conda_env:
        conda_cmd = get_conda_env(conda_env)
        cmd = f"{conda_cmd} {cmd}'"  # close the single quote opened in get_conda_env

    logging.info(f"[RUN] {step_name} | {cmd}")

    if dry_run:
        logging.info("[RUN] Dry-run enabled; command not executed.")
        return True

    # Auto-detect if this is a Blender command
    if 'blender' in cmd.lower() and not check_output:
        check_output = True
        logging.info(f"[RUN] Detected Blender command, enabling output checking for command: {cmd}")

    has_error=False
    # Execute command
    if check_output:
        return_code, output_text = _capture_command_output(cmd)
        has_error = _check_for_errors(output_text)
    else:
        # Use os.system for better compatibility with progress bars
        status = os.system(cmd)
        # os.system() returns a 16-bit wait status. The actual exit code is in the high bits.
        return_code = status >> 8

    if (return_code != 0 or has_error) and not allow_error:
        logging.error(f"[RUN] Stopping pipeline at step: {step_name}")
        if has_error:
            logging.error(f"[RUN] Error detected in output")
        sys.exit(return_code or 1)
    
    logging.info(f"[RUN] Successfully completed step: {step_name}")
    return return_code == 0 and not has_error


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_obj_class_for_id(obj_id: str, cfg: DictConfig) -> str:
    """Get object class for a given object ID."""
    assert os.path.exists(cfg.paths.obj_metadata_path), f"obj_metadata_path: {cfg.paths.obj_metadata_path} does not exist"
    # Handle relative paths by making them absolute relative to the project root
    metadata_path = cfg.paths.obj_metadata_path
    obj_metadata = load_json(metadata_path)
    return obj_metadata.get(obj_id, {}).get("obj_class", "UNKNOWN")

def download_object(obj_id: str):
    """Download a single object from Objaverse to the current directory."""
    logging.info(f"Downloading object with UID: {obj_id}")
    try:
        objects = objaverse.load_objects(uids=[obj_id])
        logging.info(f"Successfully downloaded object: {obj_id}")
        return objects.get(obj_id)
    except Exception as e:
        print(f"Failed to download object {obj_id}: {e}")  # Fallback print
        logging.error(f"Failed to download object {obj_id}: {e}")
        return None

def prepare_nerf_dataset_from_blender_output(data_dir):
    """
    Converts raw Blender output into a dataset format compatible with NeRF training.
    - Removes alpha channels from PNG images.
    - Creates `transforms.json` and empty `transforms_test.json`.
    """
    # data_dir already contains the obj_id from the pipeline
    source_path = data_dir
    train_dir = os.path.join(source_path, "train")

    train_transforms = os.path.join(source_path, "transforms_train.json")
    base_transforms = os.path.join(source_path, "transforms.json")
    
    if os.path.exists(train_transforms):
        shutil.copy(train_transforms, base_transforms)
    elif os.path.exists(base_transforms):
        shutil.copy(base_transforms, train_transforms)

    empty_test_transforms = {
        "frames": [],
        "camera_angle_x": 0.0, "camera_angle_y": 0.0
    }
    save_json(empty_test_transforms, os.path.join(source_path, "transforms_test.json"))
    if not os.path.isdir(train_dir):
        logging.warning(f"Training directory not found, skipping conversion: {train_dir}")
        return

    for filename in os.listdir(train_dir):
        if not filename.lower().endswith(".png"):
            continue
        
        img_path = os.path.join(train_dir, filename)
        try:
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img.convert('RGB').save(img_path)
        except Exception as e:
            logging.warning(f"Could not process image {img_path}: {e}")

    logging.info(f"Successfully prepared NeRF dataset for data directory: {data_dir}")


def resolve_paths(cfg: DictConfig) -> DictConfig:
    """Resolve all path configurations."""
    # Handle empty/null base_path to use current directory
    if not cfg.paths.base_path:
        cfg.paths.base_path = os.getcwd()
    if not cfg.paths.inference_results_dir:
        cfg.paths.inference_results_dir = f"inference_combined_mse_{cfg.training.feature_type}_results"
    
    # Resolve all path variables
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    return cfg


def join_path(*components):
    """Safely join path components, returning None if any component is invalid."""
    try:
        # If any component is None, return None
        if any(c is None for c in components):
            return None
        
        valid_components = [str(c) for c in components if c and str(c).strip()]
        return os.path.join(*valid_components) if valid_components else None
    except:
        return None


def get_output_paths(cfg: DictConfig, obj_id: str) -> Dict[str, str]:
    """Get all output paths for a specific object."""
    paths = {}
    
    # Blender output
    paths['data_dir'] = join_path(cfg.paths.data_dir, obj_id)
    
    # NeRF output
    paths['nerf_output'] = join_path(cfg.paths.outputs_dir, obj_id, cfg.training_3d.nerf.method)
    
    # Gaussian Splatting output
    paths['gs_output'] = join_path(cfg.paths.outputs_dir, obj_id, "gs")
    
    # Render output
    paths['render_output'] = join_path(cfg.paths.render_outputs_dir, obj_id)
    
    # Material segmentation outputs
    if cfg.material_mode == 'vlm':
        paths['vlm_base_dir'] = join_path(cfg.paths.vlm_seg_mat_sample_results_dir, obj_id)
    elif cfg.material_mode == 'neural':
        paths['neural_base_dir'] = join_path(cfg.paths.base_path, cfg.paths.inference_results_dir, cfg.segmentation.neural.result_id, obj_id)
    
    # PhysGaussian output
    paths['physgaussian_output'] = join_path(cfg.paths.physgaussian_output_dir, cfg.material_mode, obj_id)

    # Blender output
    paths['blender_output'] = join_path(cfg.paths.blender_output_dir, obj_id)
    
    # Blender scene file
    paths['blend_file_path'] = cfg.paths.blend_file_path
    
    # Blender addon paths
    paths['blender_gs_addon_path'] = cfg.paths.blender_gs_addon_path

    if not cfg.is_objaverse_object:
        logging.warning("Real data must use `disable_scene_contraction=True` and `USE_COLMAP_DATAPARSER`. Setting this automatically.")
        cfg.training_3d.nerf.disable_scene_contraction = True
        if should_use_colmap(cfg, paths):
            os.environ["USE_COLMAP_DATAPARSER"] = "1"
        assert cfg.material_mode == "neural", f"Real data must use neural material mode. You have: `{cfg.material_mode}` mode"
    return paths

def should_use_colmap(cfg: DictConfig, paths: dict) -> bool:
    """Determine if Colmap should be used for data parsing."""
    return "colmap" in os.listdir(paths['data_dir'])


def get_physics_config_path(cfg: DictConfig, obj_id: str, material_mode: str, obj_class: str) -> str:
    """Get the appropriate physics configuration path."""
    if cfg.is_objaverse_object:
        return f"{cfg.paths.physgaussian_config_dir}/objaverse/custom_{obj_class}_config.json"
    else:
        return f"{cfg.paths.physgaussian_config_dir}/real_scene/custom_{obj_id}_config.json" 


def should_use_white_bg(cfg: DictConfig, material_mode: str, obj_class: str) -> bool:
    """Determine if white background should be used for physics simulation."""
    if material_mode != 'neural' or obj_class not in cfg.physics.no_white_bg_classes:
        return cfg.physics.white_bg
    return False


def create_directories(paths: Dict[str, str]) -> None:
    """Create all necessary directories."""
    for path in paths.values():
        if path and not path.startswith('${'):
            path_obj = Path(path)
            # If it's a file path (has extension), create the parent directory
            if path_obj.suffix:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            else:
                # It's a directory path, create it directly
                path_obj.mkdir(parents=True, exist_ok=True)
    
    import json
    logging.info(f"Created directories:\n{json.dumps(paths, indent=2)}")


def get_latest_nerf_run(output_dir: str) -> Optional[str]:
    """Get the path to the latest NeRF training run directory.
    
    Returns:
        Path to the latest run directory if found, None otherwise
    """
    if not os.path.exists(output_dir):
        return None
    
    dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
            if os.path.isdir(os.path.join(output_dir, d))]
    
    return max(dirs, key=os.path.getmtime) if dirs else None





def configure_real_scene_voxelization(cfg: DictConfig) -> None:
    """Auto-configure voxelization for real scene data."""
    scene_config = load_json(os.path.join(cfg.paths.data_dir, "scene_bounds.json"))[cfg.obj_id]
    cfg.voxelization.scene_bounds = scene_config["scene_bounds"]
    cfg.voxelization.voxel_size = scene_config["voxel_size"]
    logging.info(f"Auto-configured voxelization for {cfg.obj_id}. Set scene bounds to {cfg.voxelization.scene_bounds} and voxel size to {cfg.voxelization.voxel_size}")





def validate_config(cfg: DictConfig, single_obj: bool = True) -> None:
    """Validate the configuration."""
    if cfg.obj_id is None and cfg.obj_path is None and single_obj:
        raise ValueError("Either obj_id or obj_path must be provided")
    
    if cfg.material_mode not in ['vlm', 'neural']:
        raise ValueError(f"Invalid material_mode: {cfg.material_mode}. Available modes are: vlm, neural")
    
    if cfg.material_mode == 'neural':
        # if not cfg.segmentation.neural.result_id:
        #     raise ValueError("result_id is required for neural segmentation mode")
        if not cfg.segmentation.neural.feature_type:
            raise ValueError("feature_type is required for neural segmentation mode")
    
    # Validate and resolve voxelization settings
    if cfg.voxelization.voxel_size is None and cfg.voxelization.grid_size is None:
        raise ValueError("Either voxel_size or grid_size must be provided in voxelization config")
    
    if cfg.voxelization.voxel_size is None:
        cfg.voxelization.voxel_size = 1.0 / cfg.voxelization.grid_size
    

    




def save_contextual_config(cfg: DictConfig, output_dir: str, context: str = "pipeline") -> None:
    """Save context-specific configuration for reproducibility.
    
    This function automatically determines which config sections are relevant based on the context
    and saves only those parts, making it maintainable and automatically adapting to config changes.
    
    Args:
        cfg: The full configuration
        output_dir: Directory where to save the config
        context: Context type ('blender', 'gaussian_splatting', 'voxelization', 
                any 'vlm_*' context, 'pipeline')
    """
    # Define which config sections are relevant for each context
    context_configs = {
        "blender": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "rendering": OmegaConf.to_container(cfg.data_rendering, resolve=True),
            "paths": {"blender_path": str(cfg.paths.blender_path)},
        },
        "gaussian_splatting": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "training": {"gaussian_splatting": OmegaConf.to_container(cfg.training_3d.gaussian_splatting, resolve=True)},
            "paths": {"gaussian_splatting_dir": str(cfg.paths.gaussian_splatting_dir)},
        },
        "voxelization": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "voxelization": OmegaConf.to_container(cfg.voxelization, resolve=True),
        },
        "physics_simulation": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "physics": OmegaConf.to_container(cfg.physics, resolve=True),
            "paths": {"physgaussian_dir": str(cfg.paths.physgaussian_dir)},
        },
        "blender_output_render": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "rendering": OmegaConf.to_container(cfg.output_rendering, resolve=True),
            "paths": {"blender_output_dir": str(cfg.paths.blender_output_dir)},
        },
        "blender_gs_render": {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "rendering": OmegaConf.to_container(cfg.output_rendering, resolve=True),
            "paths": {"blender_output_dir": str(cfg.paths.blender_output_dir)},
        },
    }
    
    # Handle VLM contexts with a simple base config
    if context.startswith("vlm_"):
        context_configs[context] = {
            "obj_id": cfg.obj_id,
            "obj_path": cfg.obj_path,
            "segmentation": {"vlm": {"labeling": OmegaConf.to_container(cfg.segmentation.vlm.labeling, resolve=True)}},
        }
    
    if context not in context_configs:
        raise ValueError(f"Unknown context: {context}. Available contexts: {list(context_configs.keys())}")
    
    # Get the relevant config for this context
    relevant_config = context_configs[context]
    
    # Convert to dict and save
    # If relevant_config is already a dict, use it directly; otherwise convert from OmegaConf
    if isinstance(relevant_config, dict):
        config_dict = relevant_config
    else:
        config_dict = OmegaConf.to_container(relevant_config, resolve=True)
    
    # Save as YAML
    config_name = f"{context}_config.yaml"
    config_path = Path(output_dir) / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config_dict, config_path)
    logging.info(f"Saved {context} configuration: {config_path}") 


def get_vlm_api_key(cfg: DictConfig, model_name: str) -> str:
    """Get API key from config or environment."""
    load_dotenv()
    cfg_api = cfg.segmentation.vlm.labeling.api
    if "gemini" in model_name: return os.environ.get('GEM_API_KEY') or cfg_api.gemini_api_key or ""
    elif "claude" in model_name: return os.environ.get('CLAUDE_API_KEY') or cfg_api.claude_api_key or ""
    elif "gpt" in model_name: return os.environ.get('GPT_API_KEY') or cfg_api.gpt_api_key or ""
    return ""


def get_vlm_results(cfg: DictConfig) -> Dict[str, Any]:
    """Get VLM task results as a dictionary keyed by task name."""
    results = {}
    obj_id = cfg.obj_id
    
    # VLM segmentation results
    vlm_seg_path = f"{cfg.paths.vlm_seg_results_dir}/{obj_id}/vlm_results.json"
    results['segmentation'] = load_json(vlm_seg_path) if os.path.exists(vlm_seg_path) else None
    
    # VLM segmentation critic results
    critic_path = f"{cfg.paths.vlm_seg_critic_results_dir}/{obj_id}/critic_results.json"
    results['seg_critic'] = load_json(critic_path) if os.path.exists(critic_path) else None
    
    # VLM physics sampler results  
    sampler_path = f"{cfg.paths.vlm_seg_mat_sample_results_dir}/{obj_id}/sampler_results.json"
    results['phys_sampler'] = load_json(sampler_path) if os.path.exists(sampler_path) else None
    
    # VLM parse segmentation critic results
    parse_path = f"{cfg.paths.vlm_seg_mat_sample_results_dir}/{obj_id}/parse_results.json"
    results['parse_seg_critic'] = load_json(parse_path) if os.path.exists(parse_path) else None
    
    # Final chosen VLM results (from material segmentation)
    base_dir = f"{cfg.paths.vlm_seg_mat_sample_results_dir}/{obj_id}"
    if os.path.exists(base_dir):
        sample_dirs = [d for d in os.listdir(base_dir) if d.startswith("sample_")]
        if sample_dirs:
            chosen_path = f"{base_dir}/{sample_dirs[0]}/chosen_vlm_results.json"
            results['chosen_results'] = load_json(chosen_path) if os.path.exists(chosen_path) else None
    
    return results



def load_config(config_path="../config", config_name="config"):
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

    return cfg



def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def get_material_vlm_segmentation_path(cfg: DictConfig, render_output_dir: str,
                                   paths: dict) -> Tuple[str, str]:
    """Get the path to the material segmentation."""
    sample_id = f"sample_{cfg.physics.sample_id}"
    sample_render_output_dir = os.path.join(render_output_dir, sample_id)
    point_cloud_path = os.path.join(sample_render_output_dir,
                                    "segmented_semantics.ply")
    return point_cloud_path

def get_material_neural_segmentation_path(cfg: DictConfig, render_output_dir: str,
                                   paths: dict) -> Tuple[str, str]:

    sample_output_dir = Path(paths['neural_base_dir'])
    if not cfg.is_objaverse_object:
        point_cloud_path = sample_output_dir / "world_mapped_preds.ply"
    else:
        point_cloud_path = sample_output_dir / "mapped_preds.ply"
    return point_cloud_path

def generate_material_segmentation(
        cfg: DictConfig, render_output_dir: str,
        paths: dict) -> Tuple[Optional[str], Optional[str]]:
    """Generates material segmentation for a single material sample."""
    base_sample_dir = Path(paths['vlm_base_dir'])

    material_dict_path = None
    if cfg.segmentation.material_dict_path:
        material_dict_path = Path(cfg.segmentation.material_dict_path)
    else:
        # Use configured sample_id instead of auto-detection
        sample_id = f"sample_{cfg.physics.sample_id}"
        sample_dir = base_sample_dir / sample_id
        
        if not os.path.exists(sample_dir):
            logging.error(f"Configured sample directory not found: {sample_dir}")
            return None, None
            
        material_dict_path = sample_dir / "chosen_vlm_results.json"
        logging.info(f"Using configured sample: {sample_dir}")

    if not os.path.exists(material_dict_path):
        logging.warning(f"Material dict not found, skipping: {material_dict_path}")
        return None, None

    sample_id = material_dict_path.parent.name
    logging.info(f"Processing material sample {sample_id}")

    sample_render_output_dir = os.path.join(render_output_dir, sample_id)
    os.makedirs(sample_render_output_dir, exist_ok=True)

    material_dict_copy_path = os.path.join(sample_render_output_dir,
                                           "material_dict.json")
    shutil.copy2(material_dict_path, material_dict_copy_path)
    logging.info(f"Copied material dictionary to {material_dict_copy_path}")

    segmentation_cmd = [
        "python",
        "pixie/voxel/segmentation.py",
        "--grid_feature_path",
        f"{render_output_dir}/clip_features.npz",
        "--occupancy_path",
        f"{render_output_dir}/clip_features_pc.ply",
        "--output_dir",
        sample_render_output_dir,
        "--material_dict_path",
        material_dict_copy_path,
        "--use_spatial_smoothing",
        str(cfg.segmentation.vlm.use_spatial_smoothing),
        "--overwrite",
        str(cfg.segmentation.vlm.overwrite),
        "--background_id",
        str(cfg.training.background_id),
    ]
    run_cmd(segmentation_cmd, step_name=f"SEGMENTATION_{sample_id}")

    return sample_render_output_dir

def _find_latest_inference_dir(cfg: DictConfig) -> Optional[Path]:
    """Find the latest inference results directory."""
    base_dir = Path(cfg.paths.base_path) / cfg.paths.inference_results_dir
    if not os.path.exists(base_dir):
        return None
    
    timestamp_dirs = [d for d in base_dir.iterdir() 
                     if d.is_dir() and d.name.replace('_', '').replace('-', '').isdigit()]
    return max(timestamp_dirs, key=lambda x: x.stat().st_mtime) if timestamp_dirs else None


def _get_pred_path(sample_output_dir: Path, sample_name: str) -> Path:
    """Get prediction file path."""
    if not sample_output_dir:
        return None
    return os.path.join(sample_output_dir, f"{sample_name}_pred.npy")


def _build_config_overrides(cfg: DictConfig) -> str:
    """Build config override arguments for inference script."""
    overrides = []
    
    # Essential configs that inference script needs
    if cfg.paths.base_path:
        overrides.append(f"paths.base_path={cfg.paths.base_path}")
    if cfg.training.feature_type:
        overrides.append(f"training.feature_type={cfg.training.feature_type}")
    if cfg.paths.render_outputs_dir:
        overrides.append(f"paths.render_outputs_dir={cfg.paths.render_outputs_dir}")
    overrides.append(f"training.enforce_mask_consistency={cfg.training.enforce_mask_consistency} ")
    overrides.append(f"training.inference.CONT_EPOCH={cfg.training.inference.CONT_EPOCH} ")
    overrides.append(f"training.inference.SEG_EPOCH={cfg.training.inference.SEG_EPOCH} ")

    
    return " ".join(overrides)


def generate_neural_segmentation(cfg: DictConfig, render_output_dir: str,
                                 paths: dict) -> Tuple[str, str]:
    """Generates material segmentation using a pre-trained neural network."""
    sample_name = f"sample_{cfg.physics.sample_id}"
    
    # Initialize paths
    sample_output_dir = paths['neural_base_dir']
    pred_path = _get_pred_path(sample_output_dir, sample_name)
    # Run inference if needed
    if not pred_path or not os.path.exists(pred_path):
        # Pass config overrides to inference script
        override_args = _build_config_overrides(cfg)
        inference_cmd = f"python third_party/Wavelet-Generation/trainer/inference_combined.py obj_id={cfg.obj_id} {override_args}"
        if not cfg.segmentation.neural.cache_results:
            logging.info(f"[NEURAL] Prediction path does not exist: {pred_path}. Running inference...")
            run_cmd(inference_cmd, step_name="NEURAL_INFERENCE")
        
        # Update paths to latest inference results
        latest_dir = _find_latest_inference_dir(cfg)
        if latest_dir:
            sample_output_dir = join_path(latest_dir, cfg.obj_id)
            pred_path = _get_pred_path(sample_output_dir, sample_name)
            
            # Update result_id to match the actual inference results
            cfg.segmentation.neural.result_id = latest_dir.name
            logging.info(f"[NEURAL] Updated result_id to: {cfg.segmentation.neural.result_id}")
            logging.info(f"[NEURAL] Updated prediction path to: {pred_path}")
            paths["neural_base_dir"] = sample_output_dir
        else:
            logging.error("[NEURAL] No inference results found after running inference")


    # Generate point cloud mapping
    mask_path = join_path(sample_output_dir, f"{sample_name}_mask.npy")
    point_cloud_path = join_path(sample_output_dir, "mapped_preds.ply")
    world_point_cloud_path = join_path(sample_output_dir, "world_mapped_preds.ply")
    latest_run = get_latest_nerf_run(paths['nerf_output'])
    dataparser_path = os.path.join(latest_run, "dataparser_transforms.json")

    map_cmd = (
            "python pixie/voxel/map_pred_to_coords.py "
            f"mapping.pred_path={str(pred_path)} "
            f"mapping.mask_path={str(mask_path)} "
            f"mapping.grid_feature_path={render_output_dir}/clip_features.npz "
            f"mapping.output_path={str(point_cloud_path)} "
            f"mapping.obj_id={cfg.obj_id} "
        )
    override_args = _build_config_overrides(cfg)
    map_cmd += override_args

    if not os.path.exists(point_cloud_path): 
        run_cmd(map_cmd, "MAP_PRED_TO_COORDS")
    if not cfg.is_objaverse_object and not os.path.exists(world_point_cloud_path):
        map_cmd += f" mapping.world_output_path={str(world_point_cloud_path)} "
        map_cmd += f" mapping.dataparser_path={str(dataparser_path)} "
        run_cmd(map_cmd, "MAP_PRED_TO_COORDS")

    # Create a proper sample directory path that includes the sample ID
    sample_id = f"sample_{cfg.physics.sample_id}"
    sample_render_output_dir = os.path.join(str(sample_output_dir), sample_id)
    os.makedirs(sample_render_output_dir, exist_ok=True)
    
    return sample_render_output_dir


def format_real_scene_sample(cfg: DictConfig, paths: dict):
    """Format real scene sample."""
    mat_grid = np.zeros((64, 64, 64, 4), dtype=np.float32)
    out_sample_dir = join_path(paths['render_output'], f"sample_{cfg.physics.sample_id}")
    os.makedirs(out_sample_dir, exist_ok=True)

    np.save(join_path(out_sample_dir, "material_grid.npy"), mat_grid)

def get_material_segmentation_path(cfg: DictConfig, render_output_dir: str,
                                   paths: dict) -> Tuple[str, str]:

    if cfg.material_mode == "vlm":
        return get_material_vlm_segmentation_path(cfg, render_output_dir, paths)
    elif cfg.material_mode == "neural":
        return get_material_neural_segmentation_path(cfg, render_output_dir, paths)
    else:
        raise ValueError(f"Invalid material mode: {cfg.material_mode}")
