from dataclasses import dataclass
from typing import List, Tuple

import torch
from jaxtyping import Float

from f3rm_robot.assets import get_asset_path


def sample_query_points(
    num_samples: int, mean: Tuple[float, float, float] = (0, 0, 0), std_dev: float = 0.0075
) -> Float[torch.Tensor, "num_qps 3"]:
    """
    Sample query points from a 3D gaussian with specified mean and standard deviation.
    We use the same standard deviation for the 3 dimensions. Use this to generate query points for a new Task.
    """
    assert std_dev > 0, "std_dev must be positive."

    mean = torch.tensor(mean).float()
    variance = std_dev**2
    gaussian = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.eye(3) * variance)
    query_points = gaussian.sample(torch.Size([num_samples]))
    return query_points


@dataclass(frozen=True)
class Task:
    """
    A Task which is defined by the query points, and demo embeddings. We store the demo query point features and
    density, as to get the alpha-weighted features we need the voxel size which may vary.

    The averaging of the alpha-weighted features is done upstream in the optimization script.
    """

    name: str
    query_points: Float[torch.Tensor, "num_qps 3"]

    # Features and density for each demo
    demo_features: Float[torch.Tensor, "num_demos num_qps num_channels"]
    demo_density: Float[torch.Tensor, "num_demos num_qps 1"]

    def __post_init__(self):
        assert len(self.query_points) > 0, f"Query points cannot be empty for task {self.name}"
        assert len(self.demo_features) > 0, f"Must have at least one demo for task {self.name}"
        assert (
            self.demo_features.shape[:2] == self.demo_density.shape[:2]
        ), f"Features and density must have same number of demos and query points."
        assert self.demo_density.ndim == 3 and self.demo_density.shape[-1] == 1, "Density must be 3D with 1 channel."
        assert self.demo_features.ndim == 3, "Features must be 3D."

    @property
    def num_demos(self) -> int:
        return len(self.demo_features)

    @property
    def num_query_points(self) -> int:
        return len(self.query_points)

    @property
    def num_channels(self) -> int:
        return self.demo_features.shape[-1]


def get_tasks() -> List[Task]:
    """Load all tasks from cache. Note these are for ClIP ViT-L/14@336px."""
    task_names = ["caterpillar_ear", "mug_handle", "mug_lip", "rack_place", "screwdriver_handle"]
    task_paths = [get_asset_path(f"tasks/{task_name}.pt") for task_name in task_names]
    tasks = [torch.load(task_path) for task_path in task_paths]
    return tasks
