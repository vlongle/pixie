import sys

sys.path.append("gaussian-splatting")

import argparse
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import imageio.v2 as imageio
from plyfile import PlyData, PlyElement
import re
import shutil

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *
from material_field import apply_material_field_to_simulation, transform_to_original_coordinates

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
import glob
from scipy.spatial.transform import Rotation as scipy_R

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)



def save_semantic_point_cloud(mpm_solver, gs_num, output_path, 
                              scale_origin, original_mean_pos, rotation_matrices,
                              to_original_coord=True):
    """
    Save a point cloud with material IDs, density, E, and nu as vertex data to a PLY file.
    
    Args:
        positions: Tensor of 3D point positions [N, 3]
        material_ids: Array of material IDs for each point
        density: Array of density values for each point
        E: Array of Young's modulus values for each point
        nu: Array of Poisson's ratio values for each point
        output_path: Directory to save the PLY file
        frame_number: Current frame number for filename
    """
    positions = mpm_solver.export_particle_x_to_torch()[:gs_num]
    material_ids = mpm_solver.mpm_state.particle_material.numpy()[:gs_num]
    density = mpm_solver.mpm_state.particle_density.numpy()[:gs_num]
    E = mpm_solver.mpm_model.E.numpy()[:gs_num]
    nu = mpm_solver.mpm_model.nu.numpy()[:gs_num]


    # Convert positions to numpy array
    if to_original_coord:
        positions = transform_to_original_coordinates(
            positions, scale_origin, original_mean_pos, rotation_matrices
        )
        
    positions_np = positions.detach().cpu().numpy()

    
    # Create vertex data with positions and material IDs
    vertex_data = np.zeros(positions_np.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('material_id', 'i4'),
        ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
    ])
    
    # Fill in the vertex data
    vertex_data['x'] = positions_np[:, 0]
    vertex_data['y'] = positions_np[:, 1]
    vertex_data['z'] = positions_np[:, 2]
    vertex_data['material_id'] = material_ids
    vertex_data['density'] = density
    vertex_data['E'] = E
    vertex_data['nu'] = nu
    # Create the PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Create the PLY file
    PlyData([vertex_element], text=True).write(output_path)
    
    print(f"Saved semantic point cloud to {output_path}")


def load_point_cloud(ply_path, opacity_value=0.5, sh_degree=3):
    """
    Load a point cloud from a PLY file and prepare it for simulation.
    
    Args:
        ply_path: Path to the PLY file
        opacity_value: Default opacity to assign to all points
        sh_degree: SH degree for the Gaussian model
    
    Returns:
        Dictionary with positions, default covariances, and opacities
    """

    
    # Load the point cloud
    plydata = PlyData.read(ply_path)
    vertex_element = plydata['vertex']
    

    # Extract positions
    x = vertex_element['x']
    y = vertex_element['y']
    z = vertex_element['z']
    positions = np.column_stack((x, y, z))
    
    # Optional confidence/probability of material assignment
    conf = vertex_element['conf'].astype(np.float32) if 'conf' in [p.name for p in vertex_element.properties] else np.ones(len(x), dtype=np.float32)
    
    # Check if color data exists in the PLY file
    # Get property names from the vertex element
    property_names = [p.name for p in vertex_element.properties]
    has_colors = all(color in property_names for color in ['red', 'green', 'blue'])
    
    colors_tensor=None
    if has_colors:
        # Extract colors if they exist
        red = vertex_element['red']
        green = vertex_element['green']
        blue = vertex_element['blue']
        colors = np.column_stack((red, green, blue))
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device="cuda") / 255.0
        
        # Convert colors to DC component (first spherical harmonic)
        features_dc = colors_tensor.unsqueeze(1) / 0.282095  # SH normalization factor
    else:
        # Default to gray color if no colors are present
        features_dc = torch.ones((positions.shape[0], 1, 3), device="cuda") * 0.5
    
    # Create rest of spherical harmonics
    features_rest = torch.zeros((positions.shape[0], (sh_degree + 1) ** 2 - 1, 3), device="cuda")
    
    # Combine into final SH coefficients
    shs = torch.cat([features_dc, features_rest], dim=1)
    
    # Create default covariances (small spheres)
    default_scale = 0.01  # Small default scale
    cov = torch.ones((positions.shape[0], 6), device="cuda") * default_scale
    
    # Create default opacities
    opacities = torch.ones((positions.shape[0], 1), device="cuda") * opacity_value
    
    # Create placeholder for screen points
    screen_points = torch.zeros((positions.shape[0], 3), device="cuda")
    
    positions_tensor = torch.tensor(positions, device="cuda").float()

    ## if material_id is not present, then set part_labels to material_id
    if 'part_label' not in vertex_element:
        part_labels = vertex_element['material_id']
    else:
        ## material field
        part_labels = vertex_element['part_label']

    density = vertex_element['density']
    E = vertex_element['E']
    nu = vertex_element['nu']
    material_id = vertex_element['material_id']
    
    
    return {
        "pos": positions_tensor,
        "cov3D_precomp": cov,
        "opacity": opacities,
        "shs": shs,
        "screen_points": screen_points,
        "colors": colors_tensor,

        ### material field
        "part_labels": part_labels,
        "density": density,
        "E": E,
        "nu": nu,
        "material_id": material_id,
        "conf": conf,
    }



class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def compile_video(folder, output_filename, fps=30):
    """Save a sequence of frames as a video using imageio.
    
    Args:
        folder (str): Directory containing the PNG frames
        output_filename (str): Path to save the output video
        fps (int): Frames per second
    """
    image = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):
            image.append(imageio.imread(os.path.join(folder, filename)))
    imageio.mimsave(output_filename, image, fps=fps)
    print(f"Video successfully saved to: {output_filename}")


def load_e_field_from_ply(ply_file_path, device="cuda"):
    """Load E values from a PLY file."""
    plydata = PlyData.read(ply_file_path)
    E_values = plydata['vertex']['E']
    return torch.tensor(E_values, dtype=torch.float32).to(device)


def cov3D_to_log_scales_and_quats(cov3D: torch.Tensor):
    """
    cov3D : (N,6) Gaussian symmetric covariance (σ₁₁, σ₁₂, σ₁₃, σ₂₂, σ₂₃, σ₃₃)
    returns
        log_scales : (N,3)  (log sₓ, log s_y, log s_z)
        quats      : (N,4)  (w, x, y, z)  right–handed, unit length
    """
    N = cov3D.shape[0]
    Σ = cov3D.new_zeros((N,3,3))
    Σ[:,0,0] = cov3D[:,0] ; Σ[:,0,1] = Σ[:,1,0] = cov3D[:,1]
    Σ[:,0,2] = Σ[:,2,0] = cov3D[:,2] ; Σ[:,1,1] = cov3D[:,3]
    Σ[:,1,2] = Σ[:,2,1] = cov3D[:,4] ; Σ[:,2,2] = cov3D[:,5]

    # eigen-decomposition  (ascending order)
    evals, evecs = torch.linalg.eigh(Σ)          # (N,3) , (N,3,3)

    # put the largest scale first, like gaussian-splatting does
    idx = torch.argsort(evals, dim=1, descending=True)
    evals  = evals.gather(1, idx)
    evecs  = evecs.gather(2, idx.unsqueeze(1).expand(-1,3,-1))

    scales = torch.sqrt(torch.clamp(evals, min=1e-12))
    log_scales = torch.log(scales)

    # rotation: R = eigenvectorsᵀ
    R = evecs

    # enforce right-handedness
    det = torch.det(R)
    neg = det < 0
    if neg.any():
        R[neg,:,2] *= -1

    quats_xyzw = torch.tensor(scipy_R.from_matrix(R.detach().cpu()).as_quat(), device=R.device)
    quats_wxyz = quats_xyzw[:, [3,0,1,2]]       # convert (x,y,z,w) -> (w,x,y,z)
    return log_scales, quats_wxyz

def export_gaussians_to_ply(ply_out_dir, mpm_solver, active_sh_degree, gs_num,
                            scale_origin, rotation_matrices,
                            opacity_render, shs_render, frame, preprocessing_params, original_mean_pos,
                            to_original_coord=True):

    # ── 1.  Pull positions and covariances ────────────────────────────────
    pos = mpm_solver.export_particle_x_to_torch()[:gs_num]                        # (gs_num,3)
    if to_original_coord:
        pos = transform_to_original_coordinates(
            undoshift2center111(pos, preprocessing_params["z_shift_value"]),
            scale_origin, original_mean_pos, rotation_matrices
        )

    cov_raw = mpm_solver.export_particle_cov_to_torch().view(-1, 6)[:gs_num]
    cov_world = apply_inverse_cov_rotations(
        cov_raw / (scale_origin**2), rotation_matrices)

    log_s, quat_wxyz = cov3D_to_log_scales_and_quats(cov_world.cpu())             # (gs_num,3) (gs_num,4)

    # ── 2.  Build a fully-formed GaussianModel stub ───────────────────────
    out_gs = GaussianModel(active_sh_degree)

    out_gs._xyz           = pos.detach()                                           # (gs_num,3)
    out_gs._opacity       = opacity_render[:gs_num].detach()                       # (gs_num,1)
    out_gs._features_dc   = shs_render[:gs_num, :1, :].detach()                    # (gs_num,1,3)
    out_gs._features_rest = shs_render[:gs_num, 1:, :].detach()                    # (gs_num,(Ks-1),3)
    out_gs._scaling       = log_s
    out_gs._rotation      = quat_wxyz

    # ── 3.  Write PLY using the repo's own routine ────────────────────────
    ply_path = os.path.join(ply_out_dir, f"frame_{frame:05d}.ply")
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    out_gs.save_ply(ply_path)



def save_boundary_conditions(boundary_conditions, output_path):
    """
    Save collected boundary conditions to a JSON file.
    
    Args:
        boundary_conditions: List of boundary condition dictionaries
        output_path: Directory to save the boundary conditions file
    """
    bc_file_path = os.path.join(output_path, "boundary_conditions.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_bcs = []
    for bc in boundary_conditions:
        serializable_bc = {}
        for key, value in bc.items():
            if isinstance(value, np.ndarray):
                serializable_bc[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_bc[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_bc[key] = float(value)
            else:
                serializable_bc[key] = value
        serializable_bcs.append(serializable_bc)
    
    with open(bc_file_path, 'w') as f:
        json.dump(serializable_bcs, f, indent=2)


def setup_simulation(args):
    """Initialize and setup the simulation environment"""
    # Load scene config
    print("Loading scene config...")
    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param_json(args.config)

    # Load gaussians
    print("Loading gaussians...")
    gaussians = load_checkpoint(args.model_path)
    print("loaded from", args.model_path)
    
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )
    
    return material_params, bc_params, time_params, preprocessing_params, camera_params, gaussians, pipeline, background


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_ply", action="store_true")

    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--point_cloud_path", type=str, help="Path to input point cloud PLY file")
    parser.add_argument("--optimization_output_path", type=str, default=None, help="Path to the output directory of the optimization run (containing E_field_epoch_XX.ply)")
    parser.add_argument("--replay_epoch", type=int, default=None, required=False, help="Epoch number of the E-field to load for replay. If None, uses the latest epoch found.")

    args = parser.parse_args()

    assert os.path.exists(args.model_path), "Model path does not exist!"
    assert os.path.exists(args.config), "Scene config does not exist!"
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Setup simulation
    material_params, bc_params, time_params, preprocessing_params, camera_params, gaussians, pipeline, background = setup_simulation(args)

    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    mask = params["opacity"][:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = params["pos"][mask, :]
    init_cov = params["cov3D_precomp"][mask, :]
    init_opacity = params["opacity"][mask, :]
    init_shs = params["shs"][mask, :]
    init_screen_points = params["screen_points"][mask, :]
    print(f"Particles remaining {init_pos.shape[0]} after filtering based on opacity.")

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (None, None, None, None)
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device="cuda")
        for i in range(3):
            mask &= (rotated_pos[:, i] > boundary[2 * i]) & (rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos, preprocessing_params["z_shift_value"])
    init_cov = apply_cov_rotations(init_cov, rotation_matrices) * (scale_origin ** 2)

    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params.get("particle_filling")

    if filling_params:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    print("Initializing MPM solver...")
    grid_lim = material_params["grid_lim"]
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        grid_lim / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params and filling_params.get("visualize", False):
        shs_render, opacity_render, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num], init_shs, init_cov, init_opacity, mpm_init_pos[gs_num:]
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs_render, opacity_render = init_shs, init_opacity

    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos, mpm_init_vol, mpm_init_cov,
        n_grid=material_params["n_grid"], grid_lim=grid_lim
    )
    mpm_solver.set_parameters_dict(material_params)
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    pc_params = load_point_cloud(args.point_cloud_path)
    conf_values, collected_bcs = apply_material_field_to_simulation(
        mpm_solver, pc_params, device=device,
        scale_origin=scale_origin, original_mean_pos=original_mean_pos,
        rotation_matrices=rotation_matrices,
        fix_ground=preprocessing_params["fix_ground"],
    k_smoothing_neighbors=preprocessing_params["k_smoothing_neighbors"],
        nn_distance_threshold=preprocessing_params["nn_distance_threshold"],
        only_handle_largest_cluster=preprocessing_params["only_handle_largest_cluster"],
        debug=args.debug,
    )

    # Save collected boundary conditions
    if collected_bcs and args.debug:
        save_boundary_conditions(collected_bcs, args.output_path)
        print(f"Saved {len(collected_bcs)} boundary conditions to {args.output_path}")

    if args.optimization_output_path:
        replay_epoch_to_use = args.replay_epoch
        if replay_epoch_to_use is None:
            print(f"Searching for latest epoch in {args.optimization_output_path}...")
            latest_epoch = -1
            epoch_pattern = re.compile(r"E_field_epoch_(\d+)\.ply")
            try:
                for filename in os.listdir(args.optimization_output_path):
                    match = epoch_pattern.match(filename)
                    if match and int(match.group(1)) > latest_epoch:
                        latest_epoch = int(match.group(1))
            except FileNotFoundError:
                raise FileNotFoundError(f"Optimization output path not found: {args.optimization_output_path}")

            if latest_epoch == -1:
                raise FileNotFoundError(f"No E_field_epoch_*.ply files found in {args.optimization_output_path}")
            replay_epoch_to_use = latest_epoch

        e_field_ply_path = os.path.join(args.optimization_output_path, f"E_field_epoch_{replay_epoch_to_use:02d}.ply")
        optimized_E = load_e_field_from_ply(e_field_ply_path, device=device) * 1e7
        mpm_solver.mpm_model.E = wp.from_torch(optimized_E.contiguous())
        print(f"Loaded optimized E-field from epoch {replay_epoch_to_use} into the MPM solver.")

    mpm_solver.finalize_mu_lam()

    mpm_space_viewpoint_center = torch.tensor(camera_params["mpm_space_viewpoint_center"], device="cuda").view(1, 3)
    mpm_space_vertical_upward_axis = torch.tensor(camera_params["mpm_space_vertical_upward_axis"], device="cuda").view(1, 3)
    viewpoint_center_worldspace, observant_coordinates = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center, mpm_space_vertical_upward_axis,
        rotation_matrices, scale_origin, original_mean_pos,
    )

    

    ## FOR DEBUGGING
    if args.debug:
        save_semantic_point_cloud(mpm_solver,
                                gs_num,
                                os.path.join(args.output_path, "semantic_pc.ply"),
                                scale_origin, original_mean_pos, rotation_matrices,
                                to_original_coord=True,
                                )
        save_semantic_point_cloud(mpm_solver,
                                gs_num,
                                os.path.join(args.output_path, "semantic_pc_mpm.ply"),
                                scale_origin, original_mean_pos, rotation_matrices,
                                to_original_coord=False,
                                )
    #### 

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)

    # Create organized directory structure: sample_X/frames/ and sample_X/ply_files/
    frames_dir = os.path.join(args.output_path, "frames")
    ply_files_dir = os.path.join(args.output_path, "ply_files")
    
    if os.path.exists(frames_dir): shutil.rmtree(frames_dir)
    if os.path.exists(ply_files_dir): shutil.rmtree(ply_files_dir)
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(ply_files_dir, exist_ok=True)

    for frame in tqdm(range(frame_num), desc="Simulating Frames"):
        current_camera = get_camera_view(
            args.model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )

        rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)
        pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)

        if args.render_img:
            cov3D = mpm_solver.export_particle_cov_to_torch().view(-1, 6)[:gs_num].to(device)

            pos_render = transform_to_original_coordinates(
                undoshift2center111(pos, preprocessing_params["z_shift_value"]),
                scale_origin, original_mean_pos, rotation_matrices
            )
            cov3D_render = apply_inverse_cov_rotations(cov3D / (scale_origin ** 2), rotation_matrices)

            if unselected_pos is not None:
                pos_render = torch.cat([pos_render, unselected_pos], dim=0)
                cov3D_render = torch.cat([cov3D_render, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)
            else:
                opacity, shs = opacity_render, shs_render

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos_render)
            rendering, _ = rasterize(
                means3D=pos_render,
                means2D=None,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None, rotations=None,
                cov3D_precomp=cov3D_render,
            )

            if args.save_ply:
                export_gaussians_to_ply(
                    ply_files_dir, mpm_solver,
                    gaussians.active_sh_degree, gs_num, scale_origin,
                    rotation_matrices, opacity_render, shs_render, frame, 
                    preprocessing_params, original_mean_pos,
                    to_original_coord=preprocessing_params["to_original_coord"],
                )

            cv2_img = cv2.cvtColor(rendering.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(frames_dir, f"{frame:05d}.png"), 255 * cv2_img)

        for _ in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)

    if args.render_img and args.compile_video:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if frame_files:
            fps = int(1.0 / time_params["frame_dt"])
            video_path = os.path.join(frames_dir, 'output.mp4')
            compile_video(frames_dir, video_path, fps)