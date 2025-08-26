from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
import os

from f3rm.feature_datamanager import FeatureDataManagerConfig
from f3rm.model import FeatureFieldModelConfig

def get_dataparser_config():
    """Get the appropriate dataparser config based on environment variable."""
    is_real_data = os.environ.get('USE_COLMAP_DATAPARSER', '0') == '1'
    if is_real_data:
        dataparser = ColmapDataParserConfig(train_split_fraction=0.95)
    else:
        dataparser = NerfstudioDataParserConfig(train_split_fraction=0.95)
    return dataparser

f3rm_method = MethodSpecification(
    config=TrainerConfig(
        method_name="f3rm",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=FeatureDataManagerConfig(
                feature_type="CLIP",
                dataparser=get_dataparser_config(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            # To support more GPUs, we reduce the num rays per chunk. The default was 1 << 15 which uses ~16GB of GPU
            # memory when training and using viewer. 1 << 14 uses ~12GB of GPU memory in comparison. The decrease in
            # rendering speed is not too important.
            model=FeatureFieldModelConfig(eval_num_rays_per_chunk=1 << 14),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "feature_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="F3RM with parallel NeRF training and feature field distillation.",
)
