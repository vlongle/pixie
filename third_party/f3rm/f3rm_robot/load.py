import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.pipelines.base_pipeline import Pipeline
from pytorch3d.transforms import Transform3d, matrix_to_quaternion, quaternion_to_matrix

from f3rm_robot.field_adapter import FeatureFieldAdapter


def load_nerf_to_world(dataset: Path) -> Transform3d:
    """
    Load the nerf_to_world transformation from the given dataset. i.e., the transformation that takes a point from
    the NeRF coordinate system to the world coordinate system. This includes scale+rotation+translation.
    """
    n2w_path = dataset / "nerf_to_world.json"
    if not n2w_path.exists():
        # raise ValueError(f"Could not find nerf_to_world.json in {dataset}")
        ## assume identity transform
        nerf_to_world = torch.eye(4)
        nerf_to_world = Transform3d(matrix=nerf_to_world.T)
        return nerf_to_world

    with n2w_path.open("r") as f:
        nerf_to_world_dict = json.load(f)

    nerf_to_world = nerf_to_world_dict["nerf_to_world"]
    nerf_to_world = torch.tensor(nerf_to_world).float()
    assert nerf_to_world.shape == (4, 4), f"Expected nerf_to_world to be 4x4, but got {nerf_to_world.shape}"

    # Note that Transform3D uses row vector representation (i.e., it expects the last **column** to be [0, 0, 0, 1]),
    # so we need to transpose the matrix.
    nerf_to_world = Transform3d(matrix=nerf_to_world.T)
    return nerf_to_world


def load_nerf_to_offset(camera_optimizer: CameraOptimizer) -> Transform3d:
    """
    We re-use the camera poses from a calibration run for training the feature field. However, there are might be errors
    depending on how the selfie stick is grasped. To tackle this, we optimize the camera poses during training, so we
    need to "undo" the offset in rotation and translation to account for these potential errors.

    We approximate this offset with the mean translation and rotation of the camera pose adjustments.
    """
    from nerfstudio.cameras.lie_groups import exp_map_SO3xR3

    mode = camera_optimizer.config.mode
    if mode == "off":
        # No camera pose optimization, so just return identity
        return torch.eye(4).float()

    # We only support SO3xR3 optimization mode
    if mode != "SO3xR3":
        raise NotImplementedError(f"Unsupported camera optimizer mode: {camera_optimizer.config.mode}")
    
    # Remove the check for pose_noise since it no longer exists in the newer version
    # assert camera_optimizer.pose_noise is None, "pose_noise should be None"

    # Get the camera pose adjustments, which should be a (n, 3, 4) tensor
    with torch.no_grad():
        pose_adjustments = exp_map_SO3xR3(camera_optimizer.pose_adjustment)
    assert pose_adjustments.shape[1:] == (3, 4)

    # Translation offset is just the mean translation
    translations = pose_adjustments[:, :, 3]
    translation_offset = translations.mean(dim=0)

    # Rotation offset is the mean quaternion (we can't just average rotation matrices)
    rot_matrices = pose_adjustments[:, :, :3]
    quats = matrix_to_quaternion(rot_matrices)

    # See this for details: https://stackoverflow.com/a/27410865
    accumulator = quats.T @ quats
    accumulator /= len(quats)
    eig_vals, eig_vecs = torch.linalg.eigh(accumulator)
    eig_vecs = eig_vecs[:, eig_vals.argsort(descending=True)]

    # Mean quaternion is the eigenvector corresponding to the largest eigenvalue
    mean_quat = eig_vecs[:, 0]
    mean_rot = quaternion_to_matrix(mean_quat)

    # Form the transform
    nerf_to_offset = torch.eye(4)
    nerf_to_offset[:3, :3] = mean_rot
    nerf_to_offset[:3, 3] = translation_offset

    # Transform3d uses row vector convention instead of column vector, so we need to transpose the matrix
    nerf_to_offset = Transform3d(matrix=nerf_to_offset.T)
    return nerf_to_offset


@dataclass(frozen=True)
class LoadState:
    pipeline: Pipeline
    nerf_to_world: Transform3d

    def feature_field_adapter(self) -> FeatureFieldAdapter:
        return FeatureFieldAdapter(model=self.pipeline.model, world_to_nerf=self.nerf_to_world.inverse())


def load_nerfstudio_outputs(exp_config_path: str) -> LoadState:
    """Load a Nerfstudio output for pose optimization."""
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, checkpoint_path, step = eval_setup(Path(exp_config_path))

    # Load nerf to world transformation
    nerf_to_world = load_nerf_to_world(dataset=config.data)
    nerf_to_world = nerf_to_world.to(pipeline.device)

    # Load and apply the camera optimizer offset. It's slightly confusing, as the final Nerfstudio model actually
    # lives in the 'offset' coordinate system, but we will call it nerf_to_world for simplicity.
    # nerf_to_offset = load_nerf_to_offset(camera_optimizer=pipeline.datamanager.train_camera_optimizer)
    nerf_to_offset = load_nerf_to_offset(camera_optimizer=pipeline.model.camera_optimizer)
    nerf_to_offset = nerf_to_offset.to(pipeline.device)
    offset_to_nerf = nerf_to_offset.inverse()

    # Compose nerf_to_world on top of offset_to_nerf to get the *actual* nerf_to_world transformation after
    # camera pose optimization.
    offset_to_world = offset_to_nerf.compose(nerf_to_world)
    nerf_to_world = offset_to_world
    return LoadState(pipeline, nerf_to_world)


@dataclass(frozen=True)
class ObjaverseLoadState:
    pipeline: any
    nerf_to_world: Transform3d
    
    def feature_field_adapter(self) -> FeatureFieldAdapter:
        return FeatureFieldAdapter(model=self.pipeline.model, world_to_nerf=self.nerf_to_world.inverse())


def load_nerfstudio_objaverse_outputs(exp_config_path: str) -> ObjaverseLoadState:
    """
    Load a Nerfstudio output for Objaverse synthetic data.
    
    Args:
        exp_config_path: Path to the experiment config file
        
    Returns:
        ObjaverseLoadState containing the pipeline and nerf_to_world transform
    """
    from nerfstudio.utils.eval_utils import eval_setup
    
    # Load the model using nerfstudio's eval_setup
    config, pipeline, checkpoint_path, step = eval_setup(Path(exp_config_path))
    
    # For Objaverse data, use identity transform since objects are already
    # centered and scaled nicely within a unit cube
    device = pipeline.device
    nerf_to_world_matrix = torch.eye(4, device=device)
    
    # Create the Transform3d object
    # Note: Transform3d uses row vector convention, so we transpose the matrix
    nerf_to_world = Transform3d(matrix=nerf_to_world_matrix.T)
    
    # Save the transform for future reference
    output_dir = Path(exp_config_path).parent
    with open(output_dir / "nerf_to_world.json", "w") as f:
        json.dump({
            "nerf_to_world": nerf_to_world_matrix.cpu().tolist(),
            "info": "Identity transform for pre-centered Objaverse data"
        }, f, indent=2)
    print(f"Saved nerf_to_world.json to {output_dir / 'nerf_to_world.json'}")
    
    return ObjaverseLoadState(pipeline, nerf_to_world)
