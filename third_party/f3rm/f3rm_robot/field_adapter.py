from typing import Dict

import torch
import torch.nn as nn
from jaxtyping import Float
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from pytorch3d.transforms import Transform3d

from f3rm.feature_field import FeatureFieldHeadNames
from f3rm.model import FeatureFieldModel


def get_alpha(density: Float[torch.Tensor, "*b n 1"], delta: float) -> Float[torch.Tensor, "*b n 1"]:
    """Get alpha from density using alpha compositing equation. Delta is the distance between samples."""
    delta_density = density * delta
    alpha = 1 - torch.exp(-delta_density)
    return alpha


def ray_samples_from_coords(coords: Float[torch.Tensor, "*b n 3"]) -> RaySamples:
    return RaySamples(
        frustums=Frustums(origins=coords, directions=torch.zeros_like(coords), starts=0, ends=0, pixel_area=None),
        camera_indices=torch.ones(1),
    )


class FeatureFieldAdapter(nn.Module):
    """
    Field adapter from which we can query by (x, y, z) positions to get density and features.
    We assume that the positions input are in the world frame, and convert them to the NeRF frame on the fly.
    """

    def __init__(self, model: FeatureFieldModel, world_to_nerf: Transform3d):
        super().__init__()
        self.rgb_field = model.field
        self.feature_field = model.feature_field
        # Transformation that takes points from world frame to NeRF frame
        self.world_to_nerf = world_to_nerf

    def get_ray_samples(self, world_points: Float[torch.Tensor, "*b n 3"]) -> RaySamples:
        """Get ray samples from world points by transforming into NeRF frame."""
        nerf_points = self.world_to_nerf.transform_points(world_points)
        return ray_samples_from_coords(nerf_points)

    def get_density(self, world_points: Float[torch.Tensor, "*b n 3"]) -> Float[torch.Tensor, "*b n 1"]:
        """Get density from NeRF. Use this method when you don't need RGB or feature outputs and care about speed."""
        ray_samples = self.get_ray_samples(world_points)
        density, _ = self.rgb_field.get_density(ray_samples)
        return density

    def get_alpha(self, world_points: Float[torch.Tensor, "*b n 3"], delta: float) -> Float[torch.Tensor, "*b n 1"]:
        """Get alpha from NeRF."""
        return get_alpha(self.get_density(world_points), delta)

    def get_rgb(self, world_points: Float[torch.Tensor, "*b n 3"]) -> Float[torch.Tensor, "*b n 3"]:
        """Get RGB only from NeRF"""
        ray_samples = self.get_ray_samples(world_points)
        density, density_embedding = self.rgb_field.get_density(ray_samples)
        field_outputs = self.rgb_field.get_outputs(ray_samples, density_embedding)
        rgb = field_outputs[FieldHeadNames.RGB]
        return rgb

    def forward(self, world_points: Float[torch.Tensor, "*b n 3"]) -> Dict[str, Float[torch.Tensor, "*b n c"]]:
        """Get density and features from the feature field."""
        ray_samples = self.get_ray_samples(world_points)
        density, density_embedding = self.rgb_field.get_density(ray_samples)
        # Forward through the feature field
        ff_outputs = self.feature_field(ray_samples)
        feature = ff_outputs[FeatureFieldHeadNames.FEATURE]
        outputs = {"density": density, "feature": feature}
        return outputs
