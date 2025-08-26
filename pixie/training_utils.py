"""
Shared training utilities for Wavelet-Generation training scripts.
This module contains common functions used across training_discrete.py, 
training_continuous_mse.py, and inference_combined.py to avoid code duplication.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
import yaml
from pathlib import Path
from omegaconf import DictConfig
from dotenv import load_dotenv
import wandb
from pixie.utils import save_json, set_logger
import logging

def load_normalization_ranges(cfg: DictConfig) -> DictConfig:
    """Load normalization ranges from saved statistics and update config."""
    normalization_ranges_file = Path(cfg.paths.normalization_stats_dir) / "normalization_ranges.yaml"
    
    assert normalization_ranges_file.exists(), (
        f"Normalization ranges file not found at {normalization_ranges_file}. "
        f"You must run 'python third_party/Wavelet-Generation/data_utils/inspect_ranges.py' "
        f"first to compute the actual ranges from your dataset."
    )
    
    with open(normalization_ranges_file, 'r') as f:
        ranges = yaml.safe_load(f)
    
    # Update the training config with loaded ranges
    cfg.training.density_min = ranges['density_p1']
    cfg.training.density_max = ranges['density_p99']
    cfg.training.E_min = ranges['E_p1']
    cfg.training.E_max = ranges['E_p99']
    cfg.training.nu_min = ranges['nu_p1']
    cfg.training.nu_max = ranges['nu_p99']
    
    logging.info(f"Loaded normalization ranges from {normalization_ranges_file}")
    logging.info(f"  Density: {ranges['density_p1']:.3f} - {ranges['density_p99']:.3f}")
    logging.info(f"  Young's E: {ranges['E_p1']:.3f} - {ranges['E_p99']:.3f}")
    logging.info(f"  Poisson ν: {ranges['nu_p1']:.4f} - {ranges['nu_p99']:.4f}")
    
    return cfg


def ddp_setup(rank: int, world_size: int):
    """Initialize process group for DDP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def masked_mean(x, mask, dims, eps=1e-8):
    """Mean over the entries where mask==1, keeping the batch (and channel) dims."""
    num = (x * mask).sum(dims)  # total error over fg voxels
    den = mask.sum(dims).clamp(min=1)  # #foreground voxels
    return num / (den + eps)  # eps avoids NaN when den==0


def compute_accuracy(pred_logits, target, mask=None, ignore_index: int = None):
    """Voxel‑wise accuracy excluding *ignore_index* class."""
    with torch.no_grad():
        pred = pred_logits.argmax(1)  # (N,D,H,W)
        if mask is None:
            mask = target != ignore_index
        else:
            mask = mask.bool()  # Convert float mask to boolean
        correct = (pred == target) & mask
        total = mask.sum()
        if total == 0:
            return torch.tensor(0.0, device=pred.device)
        return correct.sum().float() / total.float()


def setup_wandb(rank: int, cfg: DictConfig, project_suffix: str = ""):
    """Setup wandb logging (only on rank 0)."""
    if rank != 0:
        return None
    
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from config or environment variable
    API_KEY = cfg.training.training.wandb_api_key or os.environ.get("WANDB_API_KEY")
    try:
        wandb.login(key=API_KEY)
    except wandb.Error:
        print("wandb login failed. Skipping wandb setup.")
        return None

    # Build wandb init kwargs with optional resume capability
    wandb_kwargs = dict(
        project=f"pixie-3d{project_suffix}_{cfg.training.feature_type}",
        config={
            "learning_rate": cfg.training.training.lr,
            "batch_size": cfg.training.training.batch_size,
            "model_channels": cfg.training.training.unet_model_channels,
            "num_res_blocks": cfg.training.training.unet_num_res_blocks,
            "mix_precision": getattr(cfg.training.training, "mix_precision", False),
            "train_size_split": cfg.training.training.train_size,
        },
    )

    run_id = getattr(cfg.training.training, "wandb_run_id", None)
    if run_id:
        # Continue logging to the same dashboard
        wandb_kwargs.update({"id": run_id, "resume": "must"})

    return wandb.init(**wandb_kwargs)


def get_checkpoint_paths(cfg: DictConfig):
    """Get checkpoint paths based on feature type using config paths."""
    # Use config paths for base directories
    seg_base_dir = cfg.paths.discrete_checkpoint_dir
    cont_base_dir = cfg.paths.continuous_checkpoint_dir
    
    return seg_base_dir, cont_base_dir


def get_latest_checkpoint_dirs(seg_base_dir: str, cont_base_dir: str):
    """Get the latest timestamp directories for checkpoints."""
    import glob
    
    # Find all timestamp directories
    seg_timestamps = [d for d in os.listdir(seg_base_dir) if os.path.isdir(os.path.join(seg_base_dir, d))]
    cont_timestamps = [d for d in os.listdir(cont_base_dir) if os.path.isdir(os.path.join(cont_base_dir, d))]
    
    if not seg_timestamps or not cont_timestamps:
        raise ValueError("No timestamp directories found in checkpoint folders")
    
    # Get latest timestamp directories
    latest_seg_ts = sorted(seg_timestamps)[-1]  # Most recent timestamp
    latest_cont_ts = sorted(cont_timestamps)[-1]
    
    # Get latest checkpoints within those timestamp directories
    seg_checkpoint_dir = os.path.join(seg_base_dir, latest_seg_ts)
    cont_checkpoint_dir = os.path.join(cont_base_dir, latest_cont_ts)
    
    return seg_checkpoint_dir, cont_checkpoint_dir, latest_seg_ts, latest_cont_ts


def get_checkpoint(checkpoint_dir: str, epoch: int = -1):
    """Get a checkpoint from a directory based on epoch number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: Specific epoch to get checkpoint for. If -1, returns latest checkpoint.
    
    Returns:
        Path to the requested checkpoint, or None if not found.
    """
    import glob
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoints:
        return None
    
    # Extract epoch numbers
    epochs = [int(ckpt.split("epoch_")[-1].split(".")[0]) for ckpt in checkpoints]
    
    if epoch == -1:
        # Get latest checkpoint
        max_epoch_idx = np.argmax(epochs)
        return checkpoints[max_epoch_idx]
    else:
        # Get specific epoch checkpoint
        try:
            epoch_idx = epochs.index(epoch)
            return checkpoints[epoch_idx]
        except ValueError:
            return None


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None, rank: int = 0):
    """Load checkpoint with proper error handling."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0  # Return starting epoch
    
    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(checkpoint_path, map_location=map_location,
                            weights_only=True)
    
    def _extract_state_dict(ckpt):
        """Return bare state_dict regardless of checkpoint wrapping."""
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict"):
                if key in ckpt:
                    return ckpt[key]
        return ckpt  # assume it's already a state-dict

    try:
        model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    except RuntimeError as e:
        print(f"[Warning] Strict load failed for checkpoint – {e}. Trying non-strict.")
        model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    
    start_epoch = 0
    if isinstance(checkpoint, dict):
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
    
    if rank == 0:
        print(f"[Rank 0] Loaded checkpoint from {checkpoint_path} (starting at epoch {start_epoch})")
    
    return start_epoch


def save_train_test_splits(train_dataset, test_dataset, full_dataset, ckpt_dir: str, rank: int = 0):
    """Save train/test split object IDs for reproducibility."""
    if rank != 0:
        return
    
    train_ids = [full_dataset.obj_ids[i] for i in train_dataset.indices]
    test_ids = [full_dataset.obj_ids[i] for i in test_dataset.indices]

    print(f"Sample training object IDs: {train_ids[:5]}{'...' if len(train_ids) > 5 else ''}")
    print(f"Sample test object IDs: {test_ids[:3]}{'...' if len(test_ids) > 3 else ''}")

    save_json(train_ids, os.path.join(ckpt_dir, "train_set.json"))
    save_json(test_ids, os.path.join(ckpt_dir, "test_set.json"))
    print(f"Saved train/test split object IDs to: {ckpt_dir}")


def print_dataset_info(full_dataset, train_size: int, test_size: int, rank: int = 0):
    """Print dataset information."""
    if rank != 0:
        return
    
    print(f"\n=== Dataset Information ===")
    print(f"Total objects in dataset: {len(full_dataset)}")
    print(f"Training objects: {train_size}")
    print(f"Test objects: {test_size}")
    print(f"Train/test split ratio: {train_size/len(full_dataset):.1%}")


def extract_state_dict(checkpoint):
    """Extract state dict from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint  # assume it's already a state-dict
