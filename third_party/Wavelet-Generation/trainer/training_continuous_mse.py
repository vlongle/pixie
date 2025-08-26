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
    masked_mean, setup_wandb, save_train_test_splits, print_dataset_info
)

# -----------------------------------------------------------------------------
# Local imports – add project root to PYTHONPATH so we can reuse existing code.
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data_utils.my_data import MaterialVoxelDatasetContinuous
from models.module.diffusion_network import FeatureProjector, MyUNetModel  # we reuse network building blocks

# Use shared utility functions from pixie.training_utils


# ----------------------------------------------------------------------------
# Simple regression UNet (features -> 3 continuous channels)
# ----------------------------------------------------------------------------
class RegressionUNet(nn.Module):
    """Project CLIP features to lower dimension and predict 3 continuous channels."""

    def __init__(
        self,
        feature_channels: int,
        cond_dim: int,
        model_channels: int,
        num_res_blocks: int,
        channel_mult: Tuple[int, ...],
        attention_resolutions: Tuple[int, ...],
        grid_size: int,
        out_channels: int = 3,
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
            out_channels=out_channels,
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
        pred = self.unet(x)
        return pred  # (N,3,D,H,W)


# ----------------------------------------------------------------------------
# Training loop per process
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.training_cfg = cfg.training.training  # Keep a reference to the training-specific config
        self.material_channels = 3  # density, E, nu
        self.wandb_run = None  # will be initialized on rank 0
        self.SPATIAL = (2, 3, 4)  # dims for 5D tensors (N,C,D,H,W)

    def mse_supervision(self, mat_grid, feat_grid, mask, network, use_amp=False):
        LAMBDA_CONT = getattr(self.training_cfg, "lambda_cont", 1.0)

        mat_grid = mat_grid.to(torch.float32)
        feat_grid = feat_grid.to(torch.float32)

        pred_mat = network(feat_grid)

        fg_mask = mask.unsqueeze(1)

        diff_sq = (pred_mat - mat_grid) ** 2
        loss_per_sample = masked_mean(diff_sq, fg_mask.expand_as(diff_sq), self.SPATIAL)
        loss = (loss_per_sample.mean(1)).mean() * LAMBDA_CONT

        density_mse = masked_mean(diff_sq[:, 0:1], fg_mask, self.SPATIAL).mean()
        youngs_mse = masked_mean(diff_sq[:, 1:2], fg_mask, self.SPATIAL).mean()
        poisson_mse = masked_mean(diff_sq[:, 2:3], fg_mask, self.SPATIAL).mean()

        return {
            "loss": loss,
            "density_mse": density_mse.detach(),
            "youngs_mse": youngs_mse.detach(),
            "poisson_mse": poisson_mse.detach(),
        }

    def train(self, rank: int, world_size: int, timestamp: str):
        self.wandb_run = setup_wandb(rank, self.cfg, project_suffix="-continuous-mse")

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
            ckpt_dir = os.path.join(self.cfg.paths.continuous_checkpoint_dir, timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)

        full_dataset = MaterialVoxelDatasetContinuous(self.cfg)
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
            drop_last=True,
        )

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.training_cfg.batch_size,
            num_workers=self.training_cfg.tdata_worker,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )

        network = RegressionUNet(
            feature_channels=self.cfg.training.feature_channels,
            cond_dim=self.cfg.training.cond_dim,
            model_channels=self.training_cfg.unet_model_channels,
            num_res_blocks=self.training_cfg.unet_num_res_blocks,
            channel_mult=tuple(self.training_cfg.unet_channel_mult),
            attention_resolutions=tuple(self.training_cfg.attention_resolutions),
            grid_size=self.cfg.training.default_grid_size,
            out_channels=self.material_channels,
        ).to(rank)
        network = DDP(network, device_ids=[rank])

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.training_cfg.lr,
            betas=(self.training_cfg.beta1, self.training_cfg.beta2),
        )

        if self.training_cfg.lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.training_cfg.lr_decay_rate)
        else:
            scheduler = None

        use_amp = self.training_cfg.mix_precision
        scaler = GradScaler('cuda') if use_amp else None

        start_epoch = self.training_cfg.starting_epoch
        if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
            checkpoint = torch.load(resume_ckpt_path, map_location=lambda storage, loc: storage.cuda(rank))
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                network.module.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint.get("epoch", start_epoch - 1) + 1
            else:
                network.module.load_state_dict(checkpoint)
            if rank == 0:
                print(f"[Rank 0] Resumed training from {resume_ckpt_path} (starting at epoch {start_epoch})")

        for epoch in range(start_epoch, self.training_cfg.training_epochs + 1):
            train_sampler.set_epoch(epoch)
            network.train()
            epoch_loss = 0.0
            with tqdm(train_loader, disable=(rank != 0)) as tloader:
                tloader.set_description(f"Epoch {epoch} [Rank {rank}]")
                for feat_grid, mat_grid, mask, _ in tloader:
                    feat_grid = feat_grid.to(rank, non_blocking=True)
                    mat_grid = mat_grid.to(rank, non_blocking=True)
                    mask = mask.to(rank, non_blocking=True)

                    optimizer.zero_grad()
                    if use_amp:
                        with autocast('cuda'):
                            loss_dict = self.mse_supervision(mat_grid, feat_grid, mask, network, use_amp=True)
                            loss = loss_dict["loss"]
                        scaler.scale(loss).backward()
                        if self.training_cfg.use_gradient_clip:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(network.parameters(), self.training_cfg.gradient_clip_value)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_dict = self.mse_supervision(mat_grid, feat_grid, mask, network)
                        loss = loss_dict["loss"]
                        loss.backward()
                        if self.training_cfg.use_gradient_clip:
                            nn.utils.clip_grad_norm_(network.parameters(), self.training_cfg.gradient_clip_value)
                        optimizer.step()

                    epoch_loss += loss.item()
                    if rank == 0:
                        avg_loss = epoch_loss / (tloader.n + 1e-8)
                        tloader.set_postfix(loss=avg_loss)
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "train_density_mse": loss_dict["density_mse"].item(),
                            "train_youngs_mse": loss_dict["youngs_mse"].item(),
                            "train_poisson_mse": loss_dict["poisson_mse"].item(),
                        })

            if epoch % self.training_cfg.evaluation_interval == 0:
                test_metrics = self.evaluate(network, test_loader, rank)
                if rank == 0:
                    wandb.log({"epoch": epoch, **test_metrics})

            if rank == 0 and epoch % self.training_cfg.saving_intervals == 0:
                ckpt_fp = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": network.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }, ckpt_fp)
                if self.wandb_run is not None:
                    wandb.log({"saved_checkpoint": ckpt_fp, "epoch": epoch})

            if scheduler is not None:
                scheduler.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, network: DDP, data_loader: DataLoader, rank: int):
        network.eval()
        # Keep original per-batch metric calculation but accumulate sample-weighted sums
        unmasked_sum = torch.tensor(0.0, device=rank)
        masked_sum   = torch.tensor(0.0, device=rank)
        density_sum  = torch.tensor(0.0, device=rank)
        youngs_sum   = torch.tensor(0.0, device=rank)
        poisson_sum  = torch.tensor(0.0, device=rank)
        total_samples = torch.tensor(0.0, device=rank)

        with torch.no_grad():
            for feat_grid, mat_grid, mask, _ in data_loader:
                bs = feat_grid.size(0)

                mat_grid = mat_grid.to(rank, non_blocking=True)
                feat_grid = feat_grid.to(rank, non_blocking=True)
                mask = mask.to(rank, non_blocking=True)

                pred_mat = network(feat_grid)
                diff_sq = (pred_mat - mat_grid) ** 2  # (N,3,D,H,W)

                # Unmasked MSE for the whole batch
                unmasked_batch = diff_sq.mean().item()

                # Masked (foreground) MSE – same computation used during training
                fg_mask = mask.unsqueeze(1)
                masked_batch = masked_mean(diff_sq, fg_mask.expand_as(diff_sq), self.SPATIAL).mean().item()

                density_batch = masked_mean(diff_sq[:, 0:1], fg_mask, self.SPATIAL).mean().item()
                youngs_batch  = masked_mean(diff_sq[:, 1:2], fg_mask, self.SPATIAL).mean().item()
                poisson_batch = masked_mean(diff_sq[:, 2:3], fg_mask, self.SPATIAL).mean().item()

                # Accumulate sample-weighted sums
                unmasked_sum += unmasked_batch * bs
                masked_sum   += masked_batch * bs
                density_sum  += density_batch * bs
                youngs_sum   += youngs_batch * bs
                poisson_sum  += poisson_batch * bs
                total_samples += bs

        # Aggregate across GPUs
        if dist.is_initialized():
            for tensor in [unmasked_sum, masked_sum, density_sum, youngs_sum, poisson_sum, total_samples]:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        eps = 1e-8
        metrics = {
            "eval_unmasked_mse": (unmasked_sum / (total_samples + eps)).item(),
            "eval_masked_mse": (masked_sum   / (total_samples + eps)).item(),
            "eval_density_mse": (density_sum  / (total_samples + eps)).item(),
            "eval_youngs_mse":  (youngs_sum   / (total_samples + eps)).item(),
            "eval_poisson_mse": (poisson_sum  / (total_samples + eps)).item(),
        }
        network.train()
        if rank == 0:
            print("Evaluation – metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")
        return metrics


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