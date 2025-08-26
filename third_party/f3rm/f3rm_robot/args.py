import os
from typing import Tuple

from params_proto import ParamsProto, Proto


class OptimizationArgs(ParamsProto, cli_parse=False):
    """
    Language-Guided 6-DOF Pose Optimization for a given scene.
    """

    scene: str = Proto(help="Path to Nerfstudio scene config.yml file for the f3rm training run.")

    # Initial proposals
    voxel_size: float = Proto(0.01, help="Voxel size to discretize workspace into (in meters).")
    num_rots_per_voxel: int = Proto(8, help="Number of rotations to sample for each voxel.")
    alpha_threshold: float = Proto(0.1, help="Alpha threshold to use for marching cubes masking.")
    softmax_temperature: float = Proto(0.001, help="Temperature to use for softmax for language masking.")

    # Optimization
    num_steps: int = Proto(200, help="Number of optimization steps to use.")
    lr: float = Proto(2e-3, help="Learning rate to use for language-guided pose optimization.")
    ray_samples_per_batch: int = Proto(
        2**18, help="Number of ray samples to use per batch. Decrease if you are running out of CUDA memory."
    )

    # Pruning
    keep_proportion: float = Proto(
        0.975, help="Proportion of proposals to keep after pruning for each optimization step."
    )
    min_proposals: int = Proto(2048, help="Minimum number of proposals to keep after pruning.")
    prune_after: int = Proto(10, help="Number of optimization steps to run before pruning.")

    # Min and max bounds of the workspace in world frame with metric scale
    # min_bounds: Tuple[float, float, float] = (0.1, -0.45, 0.005)
    # max_bounds: Tuple[float, float, float] = (0.8, 0.45, 0.35)
    ## For our own scene:
    min_bounds: Tuple[float, float, float] = (-0.4, 0.05, -0.575)
    max_bounds: Tuple[float, float, float] = (-0.15, 0.3, -0.325)

    # Visualization
    visualize: bool = Proto(True, help="Whether to enable visualization of the optimization. This slows down the run.")
    viser_host: str = Proto("localhost", help="Host to use for viser visualization server.")
    viser_port: int = Proto(8012, help="Port to use for viser visualization server.")
    num_poses_to_visualize: int = Proto(10, help="Number of poses to visualize during and after optimization.")


# You can access the variables directly with OptimizationArgs.<field_name>, and do not need to instantiate an object
# of this class.
_args = OptimizationArgs


def validate_args():
    assert _args.scene, "Must specify scene config file using --scene."
    assert os.path.exists(_args.scene), f"--scene config file {_args.scene} does not exist"
    # Initial proposals
    assert 0 < _args.voxel_size < 0.1, f"--voxel_size should be between 0 and 0.1"
    assert _args.num_rots_per_voxel > 0, "--num_rots_per_voxel must be positive"
    assert 0 < _args.alpha_threshold <= 1.0, "--alpha_threshold must be between 0 and 1"
    assert _args.softmax_temperature > 0, "--softmax_temperature must be positive"
    # Optimization
    assert _args.num_steps > 0, "--num_steps must be positive"
    assert _args.lr > 0, "--lr must be positive"
    assert _args.ray_samples_per_batch > 0, "--ray_samples_per_batch must be positive"
    # Pruning
    assert 0 < _args.keep_proportion <= 1.0, "--keep_proportion must be between 0 and 1"
    assert _args.min_proposals > 0, "--min_proposals must be positive"
    assert _args.prune_after > 0, "--prune_after must be positive"
    # Check min and max bounds
    assert len(_args.min_bounds) == 3, f"--min_bounds must be a tuple of length 3, not {_args.min_bounds}"
    assert len(_args.max_bounds) == 3, f"--max_bounds must be a tuple of length 3, not {_args.max_bounds}"
    assert all(
        [min_bound < max_bound for min_bound, max_bound in zip(_args.min_bounds, _args.max_bounds)]
    ), "--min_bounds must be less than --max_bounds"
    # Visualization - try process args.visualize
    if isinstance(_args.visualize, str):
        assert _args.visualize.lower() in {"true", "false"}, "--visualize must be True or False"
        _args.visualize = _args.visualize.lower() == "true"
    assert _args.viser_port > 0, "--viser_port must be positive"
