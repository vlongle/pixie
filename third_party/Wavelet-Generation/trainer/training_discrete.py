import os
import sys
from datetime import datetime
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import wandb
import random
import glob
import json
from pathlib import Path

# Add the parent directory to sys.path to import pixie utilities
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pixie.utils import resolve_paths, validate_config
from pixie.training_utils import (
     load_normalization_ranges, ddp_setup, seed_everything,
    compute_accuracy, setup_wandb, save_train_test_splits, print_dataset_info
)



# -----------------------------------------------------------------------------
# Local imports – add project root to PYTHONPATH so we can reuse existing code.
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data_utils.my_data import MaterialSegmentationDataset  # for dataset path resolution
from models.module.diffusion_network import FeatureProjector, MyUNetModel




# ----------------------------------------------------------------------------
# Simple segmentation UNet (FeatureProjector + MyUNetModel)
# ----------------------------------------------------------------------------
class SegmentationUNet(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        cond_dim: int,
        model_channels: int,
        num_res_blocks: int,
        channel_mult: Tuple[int, ...],
        attention_resolutions: Tuple[int, ...],
        grid_size: int,
        num_classes: int,
    ):
        super().__init__()
        hidden_ch = 128 if feature_channels > cond_dim else None
        self.projector = (
           None
           if feature_channels == cond_dim
           else FeatureProjector(feature_channels, out_channels=cond_dim,
                                 hidden_channels=hidden_ch,)
       )

        self.unet = MyUNetModel(
            in_channels=cond_dim,
            model_channels=model_channels,
            out_channels=num_classes,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            spatial_size=grid_size,
            dims=3,
            activation=nn.LeakyReLU(0.02),
        )

    def forward(self, feat_grid):  # (N,feature_channels,D,H,W)
        x = feat_grid
        if self.projector is not None:
            x = self.projector(feat_grid)
        logits = self.unet(x)  # (N,num_classes,D,H,W)
        return logits


# Use shared utility functions from pixie.training_utils

# ----------------------------------------------------------------------------
# Training loop per process
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.training_cfg = cfg.training.training  # Keep a reference to the training-specific config
        self.background_id = self.cfg.training.background_id
        self.wandb_run = None

    def train(self, rank: int, world_size: int, timestamp: str):
        self.wandb_run = setup_wandb(rank, self.cfg, project_suffix="-material-seg")
        
        seed_everything(self.training_cfg.seed)

        resume_ckpt_path = self.cfg.training.training.resume_checkpoint
        resume_dir = self.cfg.training.training.resume_dir

        if resume_dir and not resume_ckpt_path:
            ckpt_candidates = glob.glob(os.path.join(resume_dir, "epoch_*.pth"))
            if ckpt_candidates:
                def _extract_epoch(fp):
                    try:
                        return int(os.path.basename(fp).split("_")[1].split(".")[0])
                    except Exception:
                        return -1
                resume_ckpt_path = max(ckpt_candidates, key=_extract_epoch)

        if resume_ckpt_path:
            ckpt_dir = os.path.dirname(resume_ckpt_path)
        else:
            ckpt_dir = os.path.join(self.cfg.paths.discrete_checkpoint_dir, timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)

        full_dataset = MaterialSegmentationDataset(self.cfg)
        train_size = int(self.training_cfg.train_size * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        print_dataset_info(full_dataset, train_size, test_size, rank)
        
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        save_train_test_splits(train_dataset, test_dataset, full_dataset, ckpt_dir, rank)
        
        if rank == 0 and self.wandb_run is not None:
            wandb.save(os.path.join(ckpt_dir, "train_set.json"))
            wandb.save(os.path.join(ckpt_dir, "test_set.json"))

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.training_cfg.batch_size,
            num_workers=self.training_cfg.tdata_worker,
            shuffle=False,
            sampler=train_sampler,
            drop_last=True
        )
        
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.training_cfg.batch_size,
            num_workers=self.training_cfg.tdata_worker,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False
        )

        model = SegmentationUNet(
            feature_channels=self.cfg.training.feature_channels,
            cond_dim=self.cfg.training.cond_dim,
            model_channels=self.training_cfg.unet_model_channels,
            num_res_blocks=self.training_cfg.unet_num_res_blocks,
            channel_mult=tuple(self.training_cfg.unet_channel_mult),
            attention_resolutions=tuple(self.training_cfg.attention_resolutions),
            grid_size=self.cfg.training.default_grid_size,
            num_classes=self.cfg.training.num_material_classes,
        ).to(rank)
        model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_cfg.lr,
            betas=(self.training_cfg.beta1, self.training_cfg.beta2),
        )

        if self.training_cfg.lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.training_cfg.lr_decay_rate)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss(ignore_index=self.background_id)

        use_amp = self.training_cfg.mix_precision
        scaler = GradScaler('cuda') if use_amp else None

        start_epoch = self.training_cfg.starting_epoch
        if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
            checkpoint = torch.load(resume_ckpt_path, map_location=lambda storage, loc: storage.cuda(rank))
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.module.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint.get("epoch", start_epoch - 1) + 1
            else:
                model.module.load_state_dict(checkpoint)
            if rank == 0:
                print(f"[Rank 0] Resumed training from {resume_ckpt_path} (starting at epoch {start_epoch})")

        for epoch in range(start_epoch, self.training_cfg.training_epochs + 1):
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            with tqdm(train_loader, disable=(rank != 0)) as tloader:
                tloader.set_description(f"Epoch {epoch} [Rank {rank}]")
                for feat_grid, mat_id, mask, _ in tloader:
                    feat_grid = feat_grid.to(rank, non_blocking=True)
                    mat_id = mat_id.to(rank, non_blocking=True)
                    mask = mask.to(rank, non_blocking=True)

                    optimizer.zero_grad()
                    if use_amp:
                        with autocast('cuda'):
                            logits = model(feat_grid)
                            loss = criterion(logits, mat_id) * mask
                            loss = loss.sum() / (mask.sum() + 1e-8)
                        scaler.scale(loss).backward()
                        if self.training_cfg.use_gradient_clip:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), self.training_cfg.gradient_clip_value)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(feat_grid)
                        loss = criterion(logits, mat_id) * mask
                        loss = loss.sum() / (mask.sum() + 1e-8)
                        loss.backward()
                        if self.training_cfg.use_gradient_clip:
                            nn.utils.clip_grad_norm_(model.parameters(), self.training_cfg.gradient_clip_value)
                        optimizer.step()

                    epoch_loss += loss.item()
                    if rank == 0:
                        avg_loss = epoch_loss / (tloader.n + 1e-8)
                        tloader.set_postfix(loss=avg_loss)
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
            if epoch % self.training_cfg.evaluation_interval == 0:
                mean_acc = self.evaluate(model, test_loader, rank)
                if rank == 0:
                    wandb.log({
                        "epoch": epoch,
                        "eval_mean_accuracy": mean_acc,
                    })

            if rank == 0 and epoch % self.training_cfg.saving_intervals == 0:
                ckpt_fp = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }, ckpt_fp)
                if self.wandb_run is not None:
                    wandb.log({"saved_checkpoint": ckpt_fp, "epoch": epoch})

            if scheduler is not None:
                scheduler.step() 

    # ----------------------- evaluation ------------------------------------
    def evaluate(self, model: DDP, data_loader: DataLoader, rank: int):
        model.eval()

        # Keep sample-level weighting (each grid counts once) but aggregate across GPUs
        total_acc_sum = torch.tensor(0.0, device=rank)
        total_samples = torch.tensor(0.0, device=rank)
        with torch.no_grad():
            for feat_grid, mat_id, mask, _ in data_loader:
                feat_grid = feat_grid.to(rank, non_blocking=True)
                mat_id = mat_id.to(rank, non_blocking=True)
                mask = mask.to(rank, non_blocking=True)
                logits = model(feat_grid)
                # Use mask for accuracy computation
                acc = compute_accuracy(logits, mat_id, mask=mask, ignore_index=self.background_id).item()
                bs = feat_grid.size(0)  # number of samples in this batch
                total_acc_sum += acc * bs
                total_samples += bs

        # Aggregate across GPUs
        if dist.is_initialized():
            dist.all_reduce(total_acc_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        mean_acc = (total_acc_sum / total_samples).item() if total_samples > 0 else 0.0
        if rank == 0:
            print(f"Evaluation – mean accuracy: {mean_acc:.4f}")
        model.train()
        return mean_acc


# ----------------------------------------------------------------------------
# Multiprocessing entry point
# ----------------------------------------------------------------------------

def run_worker(rank: int, world_size: int, cfg: DictConfig):
    ddp_setup(rank, world_size)
    # Load normalization ranges in worker process to ensure they're available
    cfg = load_normalization_ranges(cfg)
    trainer = Trainer(cfg)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.train(rank, world_size, timestamp)
    dist.destroy_process_group()
    # finish wandb on rank 0
    if rank == 0 and wandb.run is not None:
        wandb.finish()


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    print("==== Hydra Config ====")
    validate_config(cfg, single_obj=False)
    cfg = resolve_paths(cfg)
    
    print(OmegaConf.to_yaml(cfg.training))

    world_size = torch.cuda.device_count()
    mp.spawn(run_worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main() 