from typing import Dict, Optional, Tuple

import numpy as np
import tinycudann as tcnn
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor
import torch
import os
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import SceneBox

class FeatureFieldHeadNames:
    FEATURE: str = "feature"


class FeatureField(Field):
    aabb: Tensor

    def __init__(
        self,
        feature_dim: int,
        spatial_distortion: SpatialDistortion,
        aabb: Tensor,
        # Positional encoding
        use_pe: bool = True,
        pe_n_freq: int = 6,
        # Hash grid
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        # MLP head
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
        self.register_buffer("aabb", aabb)

        # Feature field has its own hash grid
        growth_factor = np.exp((np.log(max_res) - np.log(start_res)) / (num_levels - 1))
        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": start_res,
                    "per_level_scale": growth_factor,
                }
            ],
        }

        if use_pe:
            encoding_config["nested"].append(
                {
                    "otype": "Frequency",
                    "n_frequencies": pe_n_freq,
                    "n_dims_to_encode": 3,
                }
            )

        self.field = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.feature_dim,
            encoding_config=encoding_config,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )





    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")


    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Apply scene contraction
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions().detach()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        positions_flat = positions.view(-1, 3)

        # Get features from the neural network
        features = self.field(positions_flat).view(*ray_samples.frustums.directions.shape[:-1], -1)
        
        outputs[FeatureFieldHeadNames.FEATURE] = features

        return outputs



    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples)
