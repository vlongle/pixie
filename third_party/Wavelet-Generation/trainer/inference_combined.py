import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import time
import math
import json
from datetime import datetime
import logging




# Add the parent directory to sys.path to import pixie utilities
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pixie.utils import resolve_paths, validate_config, load_json, save_json
from pixie.training_utils import (
    load_normalization_ranges, ddp_setup, masked_mean,
    get_checkpoint_paths, get_latest_checkpoint_dirs, get_checkpoint,
    compute_accuracy, load_checkpoint
)
from pixie.metrics import InferenceMetrics, generate_metrics_report
from pixie.utils import set_logger

# Add the Wavelet-Generation project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


from models.module.diffusion_network import MyUNetModel, UNetModel, ConditionedUNet, FeatureProjector
from data_utils.my_data import MaterialVoxelDataset
from training_discrete import SegmentationUNet
from training_continuous_mse import RegressionUNet
from pixie.utils import set_logger



def load_test_dataset(config, obj_id=None):
    """Load and prepare test dataset with optional filtering."""
    full_dataset = MaterialVoxelDataset(config)
    
    if obj_id:
        # Filter for specific object
        filtered_indices = [i for i, sample_obj_id in enumerate(full_dataset.obj_ids) 
                          if sample_obj_id == obj_id]
        if not filtered_indices:
            return None, f"No data found for object ID: {obj_id}"
        return Subset(full_dataset, filtered_indices), None
    
    # Check for saved test split
    if config.training.inference.use_saved_test_split:
        seg_checkpoint_path = get_checkpoint_paths(config)[0]
        test_split_fp = os.path.join(os.path.dirname(seg_checkpoint_path), "test_set.json")
        assert os.path.isfile(test_split_fp), f"Test split file not found: {test_split_fp}"
        saved_test_ids = set(load_json(test_split_fp))
        filtered_indices = [i for i, oid in enumerate(full_dataset.obj_ids) 
                          if oid in saved_test_ids]
        if filtered_indices:
            return Subset(full_dataset, filtered_indices), None
    
    # Default to random split
    train_size = int(config.training.training.train_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    return test_dataset, None


def create_models(config, rank):
    """Create and return segmentation and continuous models."""
    seg_network = SegmentationUNet(
        feature_channels=config.training.feature_channels,
        cond_dim=config.training.cond_dim,
        model_channels=config.training.training.unet_model_channels,
        num_res_blocks=config.training.training.unet_num_res_blocks,
        channel_mult=tuple(config.training.training.unet_channel_mult),
        attention_resolutions=tuple(config.training.training.attention_resolutions),
        grid_size=config.training.default_grid_size,
        num_classes=config.training.num_material_classes,
    ).to(rank)
    
    cont_network = RegressionUNet(
        feature_channels=config.training.feature_channels,
        cond_dim=config.training.cond_dim,
        model_channels=config.training.training.unet_model_channels,
        num_res_blocks=config.training.training.unet_num_res_blocks,
        channel_mult=tuple(config.training.training.unet_channel_mult),
        attention_resolutions=tuple(config.training.training.attention_resolutions),
        grid_size=config.training.default_grid_size,
        out_channels=3,  # density, E, nu
    ).to(rank)
    
    return seg_network, cont_network


def process_batch(seg_network, cont_network, batch_data, config, rank, metrics, output_dir=None):
    """Process a single batch and update metrics."""
    feat_grid, mat_cont, mat_id, mask, info_batch = batch_data
    
    # Move to device
    mat_id = mat_id.to(rank)
    mat_cont = mat_cont.to(rank)
    mat_grid = torch.cat([mat_cont, mat_id.unsqueeze(1)], dim=1)
    feat_grid = feat_grid.to(rank)
    mask = mask.to(rank)
    
    batch_size = mat_grid.shape[0]
    D = mat_grid.shape[2]
    
    with torch.no_grad():
        # Get predictions
        seg_logits = seg_network(feat_grid)
        seg_pred = torch.argmax(seg_logits, dim=1)
        cont_pred = cont_network(feat_grid)
        
        # Calculate metrics
        accuracy = compute_accuracy(seg_logits, mat_id, mask=mask, 
                                  ignore_index=config.training.background_id)
        
        gt_cont = mat_grid[:, :3]
        fg_mask = mask.unsqueeze(1)
        diff_sq = (cont_pred - gt_cont) ** 2
        
        cont_mse = masked_mean(diff_sq, fg_mask.expand_as(diff_sq), dims=(2,3,4)).mean().item()
        density_mse = masked_mean(diff_sq[:, 0:1], fg_mask, dims=(2,3,4)).mean().item()
        youngs_mse = masked_mean(diff_sq[:, 1:2], fg_mask, dims=(2,3,4)).mean().item()
        poisson_mse = masked_mean(diff_sq[:, 2:3], fg_mask, dims=(2,3,4)).mean().item()
        
        # Update batch metrics
        metrics.add_batch_metrics(accuracy.item(), cont_mse, density_mse, youngs_mse, poisson_mse)
        
        # Process per-sample metrics
        for i in range(batch_size):
            obj_id = info_batch["obj_id"][i]
            
            sample_metrics = {
                "seg_acc": compute_accuracy(
                    seg_logits[i:i+1], mat_id[i:i+1], mask=mask[i:i+1],
                    ignore_index=config.training.background_id
                ).item(),
                "density_mse": masked_mean(diff_sq[i:i+1, 0:1], fg_mask[i:i+1], dims=(2,3,4)).mean().item(),
                "youngs_mse": masked_mean(diff_sq[i:i+1, 1:2], fg_mask[i:i+1], dims=(2,3,4)).mean().item(),
                "poisson_mse": masked_mean(diff_sq[i:i+1, 2:3], fg_mask[i:i+1], dims=(2,3,4)).mean().item(),
            }
            sample_metrics["cont_mse"] = (sample_metrics["density_mse"] + 
                                         sample_metrics["youngs_mse"] + 
                                         sample_metrics["poisson_mse"]) / 3.0
            
            metrics.add_sample_metrics(obj_id, sample_metrics)
            metrics.local_obj_ids.append(obj_id)
            
            # Save predictions if output directory specified
            if output_dir:
                save_predictions(
                    config, output_dir, i, obj_id, info_batch,
                    seg_pred[i], cont_pred[i], mat_grid[i], 
                    feat_grid[i], mask[i], D
                )


def save_predictions(config, output_dir, batch_idx, obj_id, info_batch,
                    seg_pred, cont_pred, gt_tensor, feat_tensor, mask, D):
    """Save prediction results for a single sample."""
    sample_id = info_batch["sample_id"][batch_idx]
    if isinstance(sample_id, torch.Tensor):
        sample_id = str(sample_id.item() if sample_id.numel() == 1 else sample_id[0].item())
    else:
        sample_id = str(sample_id)
    
    obj_out_dir = os.path.join(output_dir, obj_id)
    os.makedirs(obj_out_dir, exist_ok=True)
    
    # Create combined prediction tensor
    num_classes = config.training.num_material_classes - 1
    combined_pred = torch.zeros((3 + config.training.num_material_classes, D, D, D), 
                               device=cont_pred.device)
    combined_pred[:3] = cont_pred
    
    # One-hot encoding for segmentation
    for i in range(num_classes):
        combined_pred[3 + i] = (seg_pred == i).float()
    combined_pred[3 + num_classes] = (seg_pred == num_classes).float()  # Background
    
    
    # Save all outputs
    np.save(os.path.join(obj_out_dir, f"sample_{sample_id}_pred.npy"), 
            combined_pred.cpu().numpy())
    np.save(os.path.join(obj_out_dir, f"sample_{sample_id}_gt.npy"), 
            gt_tensor.cpu().numpy())

    # skip saving feat to save space. feat is very large ~ 800 MB
    # np.save(os.path.join(obj_out_dir, f"sample_{sample_id}_feat.npy"), 
    #         feat_tensor.cpu().numpy())
    np.save(os.path.join(obj_out_dir, f"sample_{sample_id}_mask.npy"), 
            mask.cpu().numpy())
    
    # Save info
    info_to_save = {
        "obj_id": obj_id,
        "sample_id": sample_id,
        "data_path": info_batch["data_path"][batch_idx],
        "feature_path": info_batch["feature_path"][batch_idx],
        "mask_path": info_batch["mask_path"][batch_idx]
    }
    np.save(os.path.join(obj_out_dir, f"sample_{sample_id}_info.npy"), info_to_save)











def run_inference_on_gpu(rank, world_size, config, seg_checkpoint_path, cont_checkpoint_path,
                        obj_id=None, output_dir=None, steps_factor=10, use_ddim=False,
                        *, dispersion="sem", print_table=True):
    """Main inference function for each GPU."""
    set_logger()
    # Setup
    ddp_setup(rank, world_size)
    config = load_normalization_ranges(config)
    
    # Load dataset
    test_dataset, error_msg = load_test_dataset(config, obj_id)
    if error_msg:
        if rank == 0:
            logging.error(error_msg)
        dist.destroy_process_group()
        sys.exit(1)
    
    # Create data loader
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, 
                                     rank=rank, shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.inference.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.training.inference.data_worker,
        pin_memory=True
    )
    
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Results will be saved to: {output_dir}")
        logging.info(f"Testing on {len(test_dataset)} samples")
    
    # Create models
    seg_network, cont_network = create_models(config, rank)
    
    # Load checkpoints using utility function
    load_checkpoint(seg_checkpoint_path, seg_network, rank=rank)
    load_checkpoint(cont_checkpoint_path, cont_network, rank=rank)
    
    seg_network.eval()
    cont_network.eval()
    
    # Initialize metrics
    metrics = InferenceMetrics()
    
    # Process batches
    for batch_data in tqdm(test_loader, desc=f"GPU-{rank} inference", disable=rank!=0):
        process_batch(seg_network, cont_network, batch_data, config, rank, 
                     metrics, output_dir)
    
    # Gather and report metrics
    all_metrics = metrics.gather_all_metrics(rank, world_size)
    
    if rank == 0 and output_dir:
        generate_metrics_report(config, all_metrics, output_dir, seg_checkpoint_path,
                              cont_checkpoint_path, dispersion, print_table)
    
    dist.destroy_process_group()


def main_worker(rank, world_size, config, seg_checkpoint_path, cont_checkpoint_path, 
                obj_id, output_dir, steps_factor=10, use_ddim=False, *, dispersion="sem"):
    """Worker function for multiprocessing."""
    run_inference_on_gpu(rank, world_size, config, seg_checkpoint_path, 
                        cont_checkpoint_path, obj_id, output_dir, steps_factor, 
                        use_ddim, dispersion=dispersion)


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(config: DictConfig):
    # Set up logging using the centralized set_logger function
    set_logger()
    
    logging.info("==== Hydra Config ====")
    validate_config(config, single_obj=False)
    config = resolve_paths(config)
    logging.info(OmegaConf.to_yaml(config.training))
    
    # Get checkpoint paths
    seg_base_dir, cont_base_dir = get_checkpoint_paths(config)
    seg_checkpoint_dir, cont_checkpoint_dir, latest_seg_ts, latest_cont_ts = \
        get_latest_checkpoint_dirs(seg_base_dir, cont_base_dir)
    
    logging.info(f"seg_checkpoint_dir: {seg_checkpoint_dir}")
    logging.info(f"cont_checkpoint_dir: {cont_checkpoint_dir}")
    
    
    seg_checkpoint_path = get_checkpoint(seg_checkpoint_dir, 
    epoch=config.training.inference.SEG_EPOCH)
    cont_checkpoint_path = get_checkpoint(cont_checkpoint_dir, 
    epoch=config.training.inference.CONT_EPOCH)
    
    if not seg_checkpoint_path or not cont_checkpoint_path:
        raise ValueError("Could not find checkpoints in one or both directories")
    
    logging.info(f">>> Loading segmentation checkpoint from: {seg_checkpoint_path}")
    logging.info(f">>> Loading continuous checkpoint from: {cont_checkpoint_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.paths.base_path, config.paths.inference_results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    world_size = torch.cuda.device_count()
    logging.info(f"Running inference across {world_size} GPUs")
    
    steps_factor = config.training.inference.steps_factor
    use_ddim = config.training.inference.use_ddim
    
    start_time = time.time()
    
    if world_size > 1:
        mp.spawn(
            main_worker,
            args=(world_size, config, seg_checkpoint_path, cont_checkpoint_path, 
                  config.obj_id, output_dir, steps_factor, use_ddim),
            nprocs=world_size,
            join=True
        )
    else:
        run_inference_on_gpu(0, 1, config, seg_checkpoint_path, cont_checkpoint_path,
                           config.obj_id, output_dir, steps_factor, use_ddim)
    
    total_time = time.time() - start_time
    
    # Save checkpoint info as JSON for better structure
    checkpoint_info = {
        "inference_timestamp": timestamp,
        "segmentation_model": {
            "training_timestamp": latest_seg_ts,
            "checkpoint_path": seg_checkpoint_path
        },
        "continuous_model": {
            "training_timestamp": latest_cont_ts,
            "checkpoint_path": cont_checkpoint_path
        },
        "total_time_seconds": total_time
    }
    
    checkpoint_info_path = os.path.join(output_dir, "checkpoint_info.json")
    save_json(checkpoint_info, checkpoint_info_path)
    logging.info(f">>> Created checkpoint info at: {checkpoint_info_path}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()