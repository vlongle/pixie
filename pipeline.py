import time
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple


import hydra
from omegaconf import DictConfig

from pixie.utils import (run_cmd, get_obj_class_for_id, download_object,
                         prepare_nerf_dataset_from_blender_output,
                         resolve_paths, get_output_paths,
                         get_physics_config_path, should_use_white_bg,
                         create_directories, validate_config,
                         get_latest_nerf_run, 
                         save_contextual_config,
                         get_vlm_api_key, get_vlm_results,
                         generate_material_segmentation,
                         generate_neural_segmentation,
                         set_logger,
                         configure_real_scene_voxelization,
                         format_real_scene_sample,
                         get_material_segmentation_path,
                         should_use_colmap)


def download_assets(cfg: DictConfig) -> None:
    """Download the specified 3D object from Objaverse if a local path isn't provided."""
    if cfg.obj_path is None:
        download_object(cfg.obj_id)


def render_blender_images(cfg: DictConfig, paths: dict) -> str:
    """Render images using Blender."""
    if cfg.obj_path is not None:
        obj_params = ["--obj_path", cfg.obj_path]
    else:
        obj_params = ["--obj_id", cfg.obj_id]

    blender_render_cmd_list = [
        "blender",
        "--background",
        "--python",
        "pixie/blender/generate_blendernerf_data.py",
        "--",
        *obj_params,
        "--num_images",
        cfg.data_rendering.num_images,
        "--format",
        cfg.data_rendering.format,
        "--camera_dist_min",
        cfg.data_rendering.camera_dist_min,
        "--camera_dist_max",
        cfg.data_rendering.camera_dist_max,
        "--output_dir",
        paths['data_dir'],
        "--scene_scale",
        cfg.data_rendering.scene_scale,
        "--blender_nerf_addon_path",
        cfg.paths.blender_nerf_addon_path,
    ]

    if cfg.data_rendering.transparent_bg:
        blender_render_cmd_list.append("--transparent_bg")

    blender_render_cmd = " ".join(map(str, blender_render_cmd_list))

    # Add Blender path if specified
    if cfg.paths.blender_path:
        blender_path = f'export PATH="{cfg.paths.blender_path}:$PATH";'
        blender_render_cmd = f"{blender_path} {blender_render_cmd}"

    # Check if we need to run Blender
    transforms_path = f"{paths['data_dir']}/transforms_train.json"
    if cfg.overwrite or not os.path.exists(transforms_path):
        run_cmd(blender_render_cmd, step_name="BLENDER_RENDER")
        save_contextual_config(cfg, paths['data_dir'], "blender")

    return paths['data_dir']


def train_distilled_clip_nerf(cfg: DictConfig, blender_output_dir: str,
                              paths: dict) -> Optional[str]:
    """Train the CLIP-distilled NeRF model (F3RM)."""
    prepare_nerf_dataset_from_blender_output(blender_output_dir)

    start = time.time()
    train_cmd = [
        "ns-train",
        cfg.training_3d.nerf.method,
        "--data",
        blender_output_dir,  # blender_output_dir already contains the obj_id
        "--max-num-iterations",
        cfg.training_3d.nerf.max_iterations,
        "--viewer.quit-on-train-completion",
        str(cfg.training_3d.nerf.quit_on_completion),
        "--pipeline.model.disable-scene-contraction",
        str(cfg.training_3d.nerf.disable_scene_contraction),
        "--save_only_latest_checkpoint",
        str(cfg.training_3d.nerf.save_only_latest_checkpoint),
        "--output_dir",
        cfg.paths.outputs_dir,
    ]

    # Check if we need to train
    output_dir = paths['nerf_output']
    latest_run = get_latest_nerf_run(output_dir)

    # Check if checkpoint exists
    checkpoint_exists = False
    if latest_run:
        config_path = os.path.join(latest_run, "config.yml")
        checkpoint_dir = os.path.join(latest_run, "nerfstudio_models")
        checkpoint_exists = (os.path.exists(config_path)
                             and os.path.exists(checkpoint_dir)
                             and len(os.listdir(checkpoint_dir)) > 0)

    # Train if needed
    if not latest_run or not checkpoint_exists:
        run_cmd(train_cmd, step_name="TRAIN_F3RM_RERUN")
        # Get the updated latest run after training
        latest_run = get_latest_nerf_run(output_dir)

    if not latest_run:
        raise RuntimeError(f"Failed to find NeRF training run in {output_dir}")

    config_path = os.path.join(latest_run, "config.yml")
    end = time.time()
    logging.info(f"NeRF training completed. Config path: {config_path}. Time taken: {end - start:.2f} seconds")
    return config_path


def train_gaussian_splatting(cfg: DictConfig, blender_output_dir: str,
                             paths: dict) -> None:
    """Train the Gaussian Splatting model."""
    gs_train_cmd_str = (
        f"cd {cfg.paths.gaussian_splatting_dir} && python train.py "
        f"-s {blender_output_dir} "  # blender_output_dir already contains the obj_id
        f"--iterations {cfg.training_3d.gaussian_splatting.max_iterations} "
        f"--model_path {paths['gs_output']}")

    if cfg.overwrite or not os.path.exists(
            f"{paths['gs_output']}/point_cloud"):
        run_cmd(gs_train_cmd_str, step_name="TRAIN_GS")
        save_contextual_config(cfg, paths['gs_output'], "gaussian_splatting")

    logging.info(
        f"Gaussian Splatting training completed. Model path: {paths['gs_output']}"
    )


def generate_voxels(cfg: DictConfig, config_path: str, paths: dict) -> str:
    """Generate voxel grid and CLIP features from the trained NeRF."""
    if not cfg.is_objaverse_object:
        # Auto-configure voxelization for real scene data
        configure_real_scene_voxelization(cfg)
        format_real_scene_sample(cfg, paths)

    bounds = cfg.voxelization.scene_bounds
    voxel_cmd = [
        "python", "pixie/voxel/voxelize.py",
        "--scene", config_path,
        "--output", f"{paths['render_output']}/clip_features.npz",
        "--voxel_size", str(cfg.voxelization.voxel_size),
        "--gray_threshold", str(cfg.voxelization.gray_threshold),
        "--min_x", str(bounds.x_bound[0]), "--max_x", str(bounds.x_bound[1]),
        "--min_y", str(bounds.y_bound[0]), "--max_y", str(bounds.y_bound[1]),
        "--min_z", str(bounds.z_bound[0]), "--max_z", str(bounds.z_bound[1])
    ]
    

    should_run_voxel = (
        cfg.overwrite_voxel or
        not (os.path.exists(f"{paths['render_output']}/clip_features.npz") and
             os.path.exists(f"{paths['render_output']}/clip_features_pc.ply")))

    if should_run_voxel:
        run_cmd(voxel_cmd, step_name="VOXELIZE")
        save_contextual_config(cfg, paths['render_output'], "voxelization")
    else:
        logging.info(f"Skipping voxelization because it already exists at {paths['render_output']}")

    return paths['render_output']


def run_physics_simulation(cfg: DictConfig, sample_output_dir: str,
                           point_cloud_path: str, paths: dict) -> None:
    """Runs physics simulation on a single segmented material point cloud."""
    if not sample_output_dir or not point_cloud_path:
        logging.info("No segmented sample provided to run simulation. Skipping.")
        return

    if not Path(point_cloud_path).exists():
        logging.error(f"Point cloud not found at {point_cloud_path}. Skipping simulation.")
        return

    sample_id = Path(sample_output_dir).name
    logging.info(f"Running simulation for sample {sample_id}")

    phys_config = get_physics_config_path(cfg, cfg.obj_id, cfg.material_mode,
                                          cfg.obj_class)
    
    # Use configured physgaussian output path: physgaussian_output/{obj_id}/{sample_id}
    gs_sim_out_path = f"{paths['physgaussian_output']}/{sample_id}"
    os.makedirs(gs_sim_out_path, exist_ok=True)

    phys_sim_list = [
        "xvfb-run",
        "-a",
        "python",
        "gs_simulation.py",
        "--model_path",
        paths['gs_output'],
        "--point_cloud_path",
        point_cloud_path,
        "--output_path",
        gs_sim_out_path,
        "--config",
        f"{phys_config}",  # Adjusted for cd into PhysGaussian
        "--render_img",
        "--compile_video",
    ]
    if cfg.physics.debug:
        phys_sim_list.append("--debug")

    # Add white background flag if needed
    if should_use_white_bg(cfg, cfg.material_mode, cfg.obj_class):
        phys_sim_list.append("--white_bg")
    if cfg.physics.save_ply:
        phys_sim_list.append("--save_ply")

    phys_sim_cmd = f"cd {cfg.paths.physgaussian_dir} && {' '.join(phys_sim_list)}"
    

    sim_output_exists = (Path(gs_sim_out_path) / "frames" / "output.gif").exists() or \
                        (Path(gs_sim_out_path) / "frames" / "output.mp4").exists()

    if cfg.overwrite or not sim_output_exists:
        run_cmd(phys_sim_cmd, step_name=f"PHYS_SIM_{sample_id}")
        save_contextual_config(cfg, gs_sim_out_path, "physics_simulation")
    else:
        logging.info(f"Physics simulation results already exist at {gs_sim_out_path}. Skipping.")


def run_vlm_segmentation(cfg: DictConfig, paths: dict) -> None:
    """Run VLM segmentation to generate part queries."""
    if not cfg.segmentation.vlm.labeling.enabled or cfg.material_mode != 'vlm':
        return

    logging.info("Running VLM segmentation...")
    
    model_name = cfg.segmentation.vlm.labeling.models.segmentation
    api_key = get_vlm_api_key(cfg, model_name)
    vlm_seg_output_dir = f"{cfg.paths.vlm_seg_results_dir}/{cfg.obj_id}"
    
    vlm_seg_cmd = [
        "python",
        "pixie/vlm_labeler/vlm_seg.py",
        "--obj_id",
        cfg.obj_id,
        "--obj_class",
        cfg.obj_class,
        "--output_dir",
        vlm_seg_output_dir,
        "--data_dir",
        cfg.paths.data_dir,
        "--overwrite",
        str(cfg.segmentation.vlm.labeling.overwrite),
        "--num_alternative_queries",
        str(cfg.segmentation.vlm.labeling.seg.num_alternative_queries),
        "--input_num_views",
        str(cfg.segmentation.vlm.labeling.seg.input_num_views),
        "--model_name",
        model_name,
        "--api_key",
        api_key,
    ]
    
    vlm_results_path = f"{vlm_seg_output_dir}/vlm_results.json"

    if cfg.overwrite or not os.path.exists(vlm_results_path):
        run_cmd(vlm_seg_cmd, step_name="VLM_SEG")
        save_contextual_config(cfg, vlm_seg_output_dir, "vlm_seg")
    else:
        logging.info(f"VLM segmentation results already exist at {vlm_results_path}")


def run_vlm_viz_seg_candidates(cfg: DictConfig, paths: dict) -> None:
    """Run VLM visualization of segmentation candidates."""
    if not cfg.segmentation.vlm.labeling.enabled or cfg.material_mode != 'vlm':
        return

    logging.info("Running VLM visualization of segmentation candidates...")
    
    model_name = cfg.segmentation.vlm.labeling.models.segmentation
    api_key = get_vlm_api_key(cfg, model_name)
    vlm_viz_cmd = [
        "python",
        "pixie/vlm_labeler/vlm_viz_seg_candidates.py",
        "--obj_id",
        cfg.obj_id,
        "--vlm_seg_results_dir",
        cfg.paths.vlm_seg_results_dir,
        "--render_outputs_dir",
        cfg.paths.render_outputs_dir,
        "--outputs_dir",
        cfg.paths.outputs_dir,
        "--grid_size",
        str(cfg.voxelization.grid_size),
        "--gray_threshold",
        str(cfg.voxelization.gray_threshold),
        "--overwrite",
        str(cfg.overwrite),
        "--model_name",
        model_name,
        "--api_key",
        api_key,
    ]

    vlm_seg_output_dir = f"{cfg.paths.vlm_seg_results_dir}/{cfg.obj_id}"
    
    # Check if visualization files exist
    viz_files_exist = False
    if os.path.exists(vlm_seg_output_dir):
        query_dirs = [d for d in os.listdir(vlm_seg_output_dir) if d.startswith("query_")]
        if query_dirs:
            first_query_dir = os.path.join(vlm_seg_output_dir, query_dirs[0])
            viz_files_exist = os.path.exists(os.path.join(first_query_dir, "clip.png"))

    if cfg.overwrite or not viz_files_exist:
        run_cmd(vlm_viz_cmd, step_name="VLM_VIZ_SEG_CANDIDATES")
        save_contextual_config(cfg, vlm_seg_output_dir, "vlm_viz_seg_candidates")
    else:
        logging.info("VLM visualization files already exist")


def run_vlm_seg_critic(cfg: DictConfig, paths: dict) -> None:
    """Run VLM segmentation critic to evaluate segmentation quality."""
    if not cfg.segmentation.vlm.labeling.enabled or cfg.material_mode != 'vlm':
        return

    logging.info("Running VLM segmentation critic...")
    
    model_name = cfg.segmentation.vlm.labeling.models.seg_critic
    api_key = get_vlm_api_key(cfg, model_name)
    vlm_critic_cmd = [
        "python",
        "pixie/vlm_labeler/vlm_seg_critic.py",
        "--obj_id",
        cfg.obj_id,
        "--data_dir",
        cfg.paths.data_dir,
        "--vlm_seg_results_dir",
        cfg.paths.vlm_seg_results_dir,
        "--vlm_seg_critic_results_dir",
        cfg.paths.vlm_seg_critic_results_dir,
        "--input_num_views",
        str(cfg.segmentation.vlm.labeling.critic.input_num_views),
        "--overwrite",
        str(cfg.overwrite),
        "--model_name",
        model_name,
        "--api_key",
        api_key,
    ]

    vlm_critic_output_dir = f"{cfg.paths.vlm_seg_critic_results_dir}/{cfg.obj_id}"
    run_cmd(vlm_critic_cmd, step_name="VLM_SEG_CRITIC")
    save_contextual_config(cfg, vlm_critic_output_dir, "vlm_seg_critic")


def run_vlm_phys_sampler(cfg: DictConfig, paths: dict) -> None:
    """Run VLM physics sampler to generate material samples."""
    if not cfg.segmentation.vlm.labeling.enabled or cfg.material_mode != 'vlm':
        return

    logging.info("Running VLM physics sampler...")
    
    model_name = cfg.segmentation.vlm.labeling.models.phys_sampler
    api_key = get_vlm_api_key(cfg, model_name)
    vlm_sampler_cmd = [
        "python",
        "pixie/vlm_labeler/vlm_phys_sampler.py",
        "--obj_id",
        cfg.obj_id,
        "--vlm_seg_results_dir",
        cfg.paths.vlm_seg_results_dir,
        "--vlm_seg_mat_sample_results_dir",
        cfg.paths.vlm_seg_mat_sample_results_dir,
        "--overwrite",
        str(cfg.segmentation.vlm.labeling.overwrite),
        "--num_sample_mat",
        str(cfg.segmentation.vlm.labeling.phys_sampler.num_sample_mat),
        "--model_name",
        model_name,
        "--api_key",
        api_key,
    ]

    vlm_sampler_output_dir = f"{cfg.paths.vlm_seg_mat_sample_results_dir}/{cfg.obj_id}"
    run_cmd(vlm_sampler_cmd, step_name="VLM_PHYS_SAMPLER")
    save_contextual_config(cfg, vlm_sampler_output_dir, "vlm_phys_sampler")


def run_vlm_parse_seg_critic(cfg: DictConfig, paths: dict) -> None:
    """Run VLM parse segmentation critic to finalize results."""
    if not cfg.segmentation.vlm.labeling.enabled or cfg.material_mode != 'vlm':
        return

    logging.info("Running VLM parse segmentation critic...")
    
    model_name = cfg.segmentation.vlm.labeling.models.parse_critic
    api_key = get_vlm_api_key(cfg, model_name)
    vlm_parse_cmd = [
        "python",
        "pixie/vlm_labeler/vlm_parse_seg_critic.py",
        "--obj_id",
        cfg.obj_id,
        "--vlm_seg_results_dir",
        cfg.paths.vlm_seg_results_dir,
        "--vlm_seg_critic_results_dir",
        cfg.paths.vlm_seg_critic_results_dir,
        "--vlm_seg_mat_sample_results_dir",
        cfg.paths.vlm_seg_mat_sample_results_dir,
        "--model_name",
        model_name,
        "--api_key",
        api_key,
    ]

    vlm_parse_output_dir = f"{cfg.paths.vlm_seg_mat_sample_results_dir}/{cfg.obj_id}"
    run_cmd(vlm_parse_cmd, step_name="VLM_PARSE_SEG_CRITIC")
    save_contextual_config(cfg, vlm_parse_output_dir, "vlm_parse_seg_critic")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the entire pipeline."""
    # Set up logging first
    set_logger()

    # Validate configuration
    validate_config(cfg)

    # Resolve paths and system-specific settings
    cfg = resolve_paths(cfg)

    # Auto-detect object class if not provided
    if cfg.obj_class is None and cfg.obj_id is not None:
        cfg.obj_class = get_obj_class_for_id(cfg.obj_id, cfg)

    # Get all output paths
    paths = get_output_paths(cfg, cfg.obj_id)

    # Create necessary directories
    create_directories(paths)


    start_time = time.time()

    # Download assets if needed
    if cfg.is_objaverse_object:
        download_assets(cfg)
        # Render Blender images
        render_blender_images(cfg, paths)

    data_dir = paths['data_dir']
    # Train NeRF model
    config_path = train_distilled_clip_nerf(cfg, data_dir, paths)

    if not cfg.is_objaverse_object and should_use_colmap(cfg, paths):
        data_dir = f"{data_dir}/colmap"
    # Train Gaussian Splatting model
    train_gaussian_splatting(cfg, data_dir, paths)

    # Generate voxels
    render_output_dir = generate_voxels(cfg, config_path, paths)

    # Run VLM labeling pipeline if enabled
    if cfg.material_mode == 'vlm' and cfg.segmentation.vlm.labeling.enabled:
        # Run VLM segmentation to generate part queries
        run_vlm_segmentation(cfg, paths)
        
        # Run VLM visualization of segmentation candidates
        run_vlm_viz_seg_candidates(cfg, paths)
        
        # Run VLM segmentation critic
        run_vlm_seg_critic(cfg, paths)
        
        # Run VLM physics sampler
        run_vlm_phys_sampler(cfg, paths)
        
        # Run VLM parse segmentation critic
        run_vlm_parse_seg_critic(cfg, paths)

    # Generate material segmentation
    if cfg.material_mode == 'vlm':
        sample_output_dir = generate_material_segmentation(
            cfg, render_output_dir, paths)
    elif cfg.material_mode == 'neural':
        sample_output_dir = generate_neural_segmentation(
            cfg, render_output_dir, paths)
    else:
        raise ValueError(f"Invalid material mode: {cfg.material_mode}")

    point_cloud_path = get_material_segmentation_path(cfg, render_output_dir, paths)
    # Run physics simulation
    run_physics_simulation(cfg, sample_output_dir, str(point_cloud_path), paths)

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
