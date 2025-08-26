import time
import os
import logging
import shutil
import glob
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig

from pixie.utils import (run_cmd, resolve_paths,
                         get_output_paths, create_directories, validate_config,
                         get_latest_nerf_run, save_contextual_config,
                         generate_material_segmentation,
                         generate_neural_segmentation,
                         get_material_vlm_segmentation_path,
                         get_material_neural_segmentation_path,
                         set_logger)
from pixie.viz_utils import compile_video
import cv2

def get_video_fps(video_path: str) -> float:
    """Extracts frames-per-second (FPS) from a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def move_files_to_folder(src_dir: Path, dst_dir: Path) -> None:
    """Move all files from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    for file_path in src_dir.iterdir():
        if file_path.is_file():
            shutil.move(str(file_path), str(dst_dir / file_path.name))

    # Remove empty source directory
    if src_dir.exists() and not any(src_dir.iterdir()):
        src_dir.rmdir()


def render_nerf_model(cfg: DictConfig, paths: dict) -> str:
    """Render RGB images from the trained NeRF model."""
    # Get the latest NeRF training run
    nerf_output_dir = paths['nerf_output']
    latest_run = get_latest_nerf_run(nerf_output_dir)

    if not latest_run:
        raise RuntimeError(
            f"No NeRF training run found in {nerf_output_dir}. Please run the pipeline first."
        )

    config_path = os.path.join(latest_run, "config.yml")
    if not os.path.exists(config_path):
        raise RuntimeError(f"NeRF config file not found at {config_path}")

    # Create render output directory
    render_output_dir = f"{paths['render_output']}/rgb_renders"
    os.makedirs(render_output_dir, exist_ok=True)

    # Check if we need to render (check for existing rendered images in subdirectories)
    rendered_images_exist = any(glob.glob(f"{render_output_dir}/*/*/*.{ext}") 
                               for ext in ['png', 'jpg', 'jpeg'])

    # Render RGB images if needed
    if cfg.overwrite or not rendered_images_exist:
        render_cmd = [
            "ns-render",
            "dataset",
            "--load-config",
            config_path,
            "--output-path",
            render_output_dir,
            "--rendered-output-names",
            cfg.output_rendering.nerf_render.rendered_output_names,
            "--split",
            cfg.output_rendering.nerf_render.split,
        ]

        run_cmd(render_cmd, step_name="RENDER_RGB")
        logging.info(
            f"RGB rendering completed. Output directory: {render_output_dir}")
    else:
        logging.info(
            f"RGB renders already exist at {render_output_dir}. Skipping rendering."
        )

    return render_output_dir

def render_blender_gs(cfg: DictConfig, paths: dict) -> str:
    """Render Gaussian Splatting outputs using Blender."""
    blender_output_dir = paths['blender_output']
    render_cfg = cfg.output_rendering.blender_render_gs
    
    # Get the ply_dir from physgaussian_output
    ply_dir = os.path.join(paths['physgaussian_output'], f"sample_{cfg.physics.sample_id}", "ply_files")
    assert os.path.exists(ply_dir), f"PLY directory does not exist: {ply_dir}"
    
    # Create output directory for GS renders
    gs_output_dir = os.path.join(blender_output_dir, "gs_renders")
    os.makedirs(gs_output_dir, exist_ok=True)
    
    # Check if video already exists
    video_output = os.path.join(gs_output_dir, f"output.mp4")
    if os.path.exists(video_output) and not cfg.overwrite:
        logging.info(f"GS render video already exists at {video_output}. Skipping rendering.")
        return gs_output_dir
    
    # Build command to run render_gs.py
    cmd = [
        "blender", "-b", "-P", "pixie/blender/render_gs.py", "--",
        "--obj_id", cfg.obj_id,
        "--ply_dir", ply_dir,
        "--output_dir", gs_output_dir,
        "--data_dir", paths['data_dir'],
        "--blend_file_path", paths['blend_file_path'],
        "--blender_gs_addon_path", paths['blender_gs_addon_path']
    ]
    
    # Add place_on_ground flag if enabled
    if render_cfg.get('place_on_ground', False):
        cmd.append("--place_on_ground")
    
    # Add optional arguments based on render config
    if render_cfg.get('rotate_around') is not None:
        cmd.extend(["--rotate_around", str(render_cfg.rotate_around)])
    
    if render_cfg.get('camera_id') is not None:
        cmd.extend(["--camera_id", str(render_cfg.camera_id)])
        
    if render_cfg.get('focal_length_ratio') is not None:
        cmd.extend(["--focal_length_ratio", str(render_cfg.focal_length_ratio)])
        
    if render_cfg.get('transparent', False):
        cmd.append("--transparent")
        
    if render_cfg.get('resolution_x') is not None:
        cmd.extend(["--resolution_x", str(render_cfg.resolution_x)])
        
    if render_cfg.get('resolution_y') is not None:
        cmd.extend(["--resolution_y", str(render_cfg.resolution_y)])
        
    if render_cfg.get('cycles_samples') is not None:
        cmd.extend(["--cycles_samples", str(render_cfg.cycles_samples)])
        
    if render_cfg.get('num_renders') is not None:
        cmd.extend(["--num_renders", str(render_cfg.num_renders)])
        
    if render_cfg.get('start_frame', 0) != 0:
        cmd.extend(["--start_frame", str(render_cfg.start_frame)])
        
    if render_cfg.get('is_dropping', False):
        cmd.append("--is_dropping")
        
    if render_cfg.get('save_blend', False):
        cmd.append("--save_blend")
        
    if render_cfg.get('bg_color') is not None:
        cmd.extend(["--bg_color"] + [str(c) for c in render_cfg.bg_color])
        
    # Add init_xyz and xyz_rotation
    init_xyz = render_cfg.get('init_xyz', [0.0, 0.0, 0.0])
    cmd.extend(["--init_xyz"] + [str(x) for x in init_xyz])
    
    xyz_rotation = render_cfg.get('xyz_rotation', [0.0, 0.0, 0.0])
    cmd.extend(["--xyz_rotation"] + [str(r) for r in xyz_rotation])
        
    if cfg.overwrite:
        cmd.append("--overwrite")
    
    # Compile frames to video
    frames_dir = Path(gs_output_dir)

    # Run the rendering command
    run_cmd(cmd, step_name="RENDER_BLENDER_GS")
    save_contextual_config(cfg, blender_output_dir, context="blender_gs_render")
    
    assert frames_dir.exists(), f"Frames directory {frames_dir} does not exist"
    assert any(frames_dir.glob('*.png')), f"No frames found in {frames_dir}"
    logging.info(f"Compiling frames to video: {video_output}")

    # Get FPS from physics video or use default
    physics_video_path = os.path.join(paths["physgaussian_output"],
                                      f"sample_{cfg.physics.sample_id}",
                                      "frames", "output.mp4")
    fps = render_cfg.default_fps
    if os.path.exists(physics_video_path):
        fps = get_video_fps(physics_video_path)
    logging.info(f"FPS: {fps}")
    compile_video(frames_dir, video_output, fps)
    
    logging.info(f"Gaussian Splatting rendering completed. Output directory: {gs_output_dir}")
    return gs_output_dir


def render_blender_glb(cfg: DictConfig, paths: dict) -> str:
    blender_output_dir = paths['blender_output']
    render_cfg = cfg.output_rendering.blender_render_glb

    ## get the point_cloud_path which contains the material physics predictions
    if cfg.material_mode == 'vlm':
        point_cloud_path = get_material_vlm_segmentation_path(
            cfg, paths['render_output'], paths)
    elif cfg.material_mode == 'neural':
        point_cloud_path = get_material_neural_segmentation_path(
            cfg, paths['render_output'], paths)
    else:
        raise ValueError(f"Invalid material mode: {cfg.material_mode}")

    for feature, cmap in zip(render_cfg.features, render_cfg.cmaps):
        cmd = (
                    f"blender -b -P pixie/blender/apply_feature_colors.py -- "
                    f"--obj_ids {cfg.obj_id} "
                    f"--feature {feature} "
                    f"--pred_ply {point_cloud_path} "
                    f"--output_dir {blender_output_dir} "
                    f"--colormap {cmap} "
                    f"--data_dir {paths['data_dir']} "
                    f"--blend_file_path {paths['blend_file_path']} "
                    f"--overwrite "
                )
        if render_cfg.rotate_video:
            cmd += f"--rotate_video "
            if render_cfg.views:
                cmd += f"--views {render_cfg.views} "
        if render_cfg.focal_length:
            cmd += f"--focal_length {render_cfg.focal_length} "
        if render_cfg.camera_id:
            cmd += f"--camera_id {render_cfg.camera_id} "

        frames_dir = Path(blender_output_dir) / feature
        video_output = frames_dir / f"{feature}.mp4"
        if os.path.exists(video_output):
            continue
        run_cmd(cmd, step_name="RENDER_BLENDER")
        save_contextual_config(cfg, blender_output_dir, context="blender_output_render")
        
        # # Compile frames to video if rotate_video is enabled
        if render_cfg.rotate_video:
            # Frames are saved in blender_output_dir/feature subdirectories
            assert frames_dir.exists(), f"Frames directory {frames_dir} does not exist"
            assert any(frames_dir.glob('*.png')), f"No frames found in {frames_dir}"
            logging.info(f"Compiling frames to video: {video_output}")

            video_path = os.path.join(paths["physgaussian_output"],
                                      f"sample_{cfg.physics.sample_id}",
                                      "frames", "output.mp4")
            fps = cfg.output_rendering.blender_render_glb.default_fps
            if os.path.exists(video_path):
                fps = get_video_fps(video_path)
            logging.info(f"FPS: {fps}")
            compile_video(frames_dir, video_output, fps)

    return blender_output_dir


def render_gs_model(cfg: DictConfig, paths: dict) -> str:
    render_gs_cmd = [
        "python", "third_party/PhysGaussian/gaussian-splatting/render.py",
        "--model_path", paths['gs_output'],
    ]
    output_dir = os.path.join(paths['gs_output'], "test")
    if not os.path.exists(output_dir):
        run_cmd(render_gs_cmd, step_name="RENDER_GS")
    return output_dir

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to render images from the trained NeRF model."""
    # Set up logging first
    set_logger()

    # Validate configuration
    validate_config(cfg)

    # Resolve paths and system-specific settings
    cfg = resolve_paths(cfg)



    # Get all output paths
    paths = get_output_paths(cfg, cfg.obj_id)

    # Create necessary directories
    create_directories(paths)

    start_time = time.time()

    if 'nerf' in cfg.output_rendering.render:
        # Render RGB images from test set
        output_dir = render_nerf_model(cfg, paths)
        logging.info(f"[NERF] Output directory: {output_dir}")

    if 'gs' in cfg.output_rendering.render:
        output_dir = render_gs_model(cfg, paths)
        logging.info(f"[GS] Output directory: {output_dir}")

    if 'blender_glb' in cfg.output_rendering.render:
        output_dir = render_blender_glb(cfg, paths)
        logging.info(f"[BLENDER_GLB] Output directory: {output_dir}")

    if 'blender_gs' in cfg.output_rendering.render:
        output_dir = render_blender_gs(cfg, paths)
        logging.info(f"[BLENDER_GS] Output directory: {output_dir}")
    end_time = time.time()
    logging.info(f"Rendering completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
