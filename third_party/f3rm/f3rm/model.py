from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from torch.nn import Parameter

from f3rm.feature_field import FeatureField, FeatureFieldHeadNames
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer


@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    # Weighing for the feature loss
    feat_loss_weight: float = 1e-3
    # Feature Field Positional Encoding
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    # Feature Field Hash Grid
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    # Feature Field MLP Head
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2



@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def clip(self):
        from f3rm.features.clip import load
        from f3rm.features.clip_extract import CLIPArgs

        CONSOLE.print(f"Loading CLIP {CLIPArgs.model_name} for viewer")
        model, _ = load(CLIPArgs.model_name, device=self.device)
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        from f3rm.features.clip import tokenize

        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        if is_positive:
            self.positives = texts
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed /= embed.norm(dim=-1, keepdim=True)
            self.pos_embed = embed
        else:
            self.negatives = texts
            # We don't average the negatives as we compute pair-wise softmax
            embed /= embed.norm(dim=-1, keepdim=True)
            self.neg_embed = embed

    @property
    def has_positives(self) -> bool:
        return self.positives and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return self.negatives and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils()


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    renderer_feature: FeatureRenderer

    def populate_modules(self):
        super().populate_modules()

        # Create feature field
        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError(f"Feature dimensionality must be positive, not {feature_dim}")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            aabb=self.scene_box.aabb,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
        )
        self.renderer_feature = FeatureRenderer()
        self.setup_gui()

    def set_language_queries(self, positives: Optional[List[str]] = None, negatives: Optional[List[str]] = None):
        """Set language queries for the model.
        
        Args:
            positives: List of positive text queries
            negatives: List of negative text queries
    
        """
        if positives is not None:
            viewer_utils.handle_language_queries(positives, is_positive=True)
        if negatives is not None:
            viewer_utils.handle_language_queries(negatives, is_positive=False)

    def setup_gui(self):
        viewer_utils.device = self.kwargs["device"]
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        # if self.kwargs["metadata"]["feature_type"] != "CLIP":
        #     return
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=viewer_utils.softmax_temp,
            cb_hook=lambda elem: viewer_utils.update_softmax_temp(elem.value),
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        ## transmittance weights in front-to-back alpha blending computed
        ## from predicted densities: field_outputs[FieldHeadNames.DENSITY]
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        ### do volumetric rendering via ray marching basically
        ### pixel_rgb = sum(weights_i * rgb_i, axis=1)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Feature outputs
        ff_outputs = self.feature_field(ray_samples)
        ### features is also rendered volumetrically
        ### features = sum(weights_i * features_i, axis=1)
        features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Compute feature error
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Compute feature loss
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)
        return loss_dict

    

    def get_sim(self, clip_features):
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not viewer_utils.has_negatives:
            sims = clip_features @ viewer_utils.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            return sims

        print(">>> Similarity for positives:", viewer_utils.positives, "and negatives:", viewer_utils.negatives)
        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / viewer_utils.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        return sims

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not viewer_utils.has_negatives:
            sims = clip_features @ viewer_utils.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            outputs["similarity"] = sims
            return outputs

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / viewer_utils.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        outputs["similarity"] = sims
        return outputs