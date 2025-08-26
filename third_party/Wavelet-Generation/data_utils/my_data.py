import torch
import os
import numpy as np
import socket
import yaml
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from typing import Tuple
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from pixie.utils import get_obj_class_for_id, load_json
from pixie.training_utils import load_normalization_ranges
import logging
# --------------------------------------------------
# dataset ----------------------------------------------------------------
# --------------------------------------------------
class MaterialVoxelDataset(torch.utils.data.Dataset):
    """
    Each sample directory must contain
        ├─ clip_features_features.npy      (grid_size,grid_size,grid_size,feature_channels) float32
        ├─ clip_features_mask.npy          (grid_size,grid_size,grid_size) float32
        └─ sample_0
           └─ material_grid.npy            (grid_size,grid_size,grid_size,material_channels)   float32
           # └─ mask.npy                     (grid_size,grid_size,grid_size)     float32 (optional) - OLD REFERENCE
    """

    def __init__(self, cfg):
        """
        Args
        ----
        cfg : omegaconf.DictConfig
              needs the following keys
                  .data_dir
                  .sample_id
                  .default_grid_size
                  .num_material_classes
                  .density_min / .density_max   (after log10)
                  .E_min / .E_max               (after log10)
                  .nu_min / .nu_max
        """
        self.cfg = cfg
        self.to_normalize = self.cfg.training.to_normalize
        self.feature_type = self.cfg.training.feature_type
        self.target_obj_classes = self.cfg.training.target_obj_classes
        self.problematic_objects = self._load_problematic_objects()
        self.data_files, self.feature_files, self.mask_files, self.obj_ids = self._collect_files()
        logging.info(f"[DATASET] Loaded {len(self.data_files)} data files.")

    # ---------- I/O helpers --------------------------------------------
    def _load_problematic_objects(self):
        """Load list of problematic object IDs to skip."""
        problematic_file = Path(self.cfg.paths.normalization_stats_dir) / "problematic_objects.json"
        if problematic_file.exists():
            problematic_objects = load_json(problematic_file)
            logging.warning(f"Loaded {len(problematic_objects)} problematic objects to skip")
            return set(problematic_objects)
        return set()

    def _collect_files(self):
        data_files, feat_files, mask_files, obj_ids = [], [], [], []
        D = self.cfg.training.default_grid_size
        C_feat = self.cfg.training.feature_channels
        initial_C_mat = self.cfg.training.in_material_channels
        logging.info(f"[DATASET] Loading data from {self.cfg.paths.render_outputs_dir}")
        for obj_id in os.listdir(self.cfg.paths.render_outputs_dir):
            # Skip problematic objects
            if obj_id in self.problematic_objects:
                logging.warning(f"Skipping {obj_id} because it is in the problematic objects list")
                continue
                
            obj_class = get_obj_class_for_id(obj_id, self.cfg)
            if self.target_obj_classes is not None and obj_class not in self.target_obj_classes:
                continue

            if self.feature_type == "clip":
                feat_fp = f"{self.cfg.paths.render_outputs_dir}/{obj_id}/clip_features_features.npy"
            elif self.feature_type == "rgb":
                feat_fp = f"{self.cfg.paths.render_outputs_dir}/{obj_id}/clip_features_rgb.npy"
            elif self.feature_type == "occupancy":
                feat_fp = f"{self.cfg.paths.render_outputs_dir}/{obj_id}/sample_{self.cfg.training.sample_id}/mask.npy"
            else:
                raise ValueError(f"Invalid feature type: {self.feature_type}")
            mat_fp  = (
                f"{self.cfg.paths.render_outputs_dir}/{obj_id}/sample_{self.cfg.training.sample_id}/"
                f"material_grid.npy"
            )
            # mask_fp = (
            #     f"{self.cfg.paths.render_outputs_dir}/{obj_id}/sample_{self.cfg.training.sample_id}/"
            #     f"mask.npy"
            # )
            mask_fp = (
                f"{self.cfg.paths.render_outputs_dir}/{obj_id}/clip_features_mask.npy"
            )
            if not os.path.exists(feat_fp):
                logging.warning(f"Skipping {obj_id} because {feat_fp} does not exist")
                continue
            if not os.path.exists(mat_fp):
                logging.warning(f"Skipping {obj_id} because {mat_fp} does not exist")
                continue
            
            # ---------------- Material-ID sanity check -----------------
            try:
                mat_chn = np.load(mat_fp, mmap_mode="r")[..., -1]  # (D,D,D) int - last channel is material_id
                max_id  = mat_chn.max()
                min_id  = mat_chn.min()
                if not (0 <= min_id < self.cfg.training.num_material_classes) or max_id >= self.cfg.training.num_material_classes:
                    logging.warning(f"[Data warning] Skipping {obj_id}: material_id outside valid range Your range: (min {min_id}, max {max_id}). Valid range: (0, {self.cfg.training.num_material_classes - 1})")
                    continue  # invalid labels → skip sample
                    # raise ValueError(f"[Data warning] Skipping {obj_id}: material_id outside valid range Your range: (min {min_id}, max {max_id}). Valid range: (0, {self.cfg.training.num_material_classes - 1})")
            except Exception as e:
                logging.warning(f"[Data warning] Could not validate material_id for {obj_id}: {e}. Skipping.")
                continue

            mat_shape  = np.load(mat_fp,  mmap_mode='r').shape  # (D,H,W,material_channels) ?
            feat_shape = np.load(feat_fp, mmap_mode='r').shape  # (D,H,W,feature_channels) ?
            if len(feat_shape) == 3:
                feat_shape = (D, D, D, 1)

            if not (
                mat_shape == (D, D, D, initial_C_mat)
                and feat_shape == (D, D, D, C_feat)
            ):
                logging.warning(f"[Data warning] Skipping {obj_id}: mat_shape: {mat_shape} != ({D, D, D, initial_C_mat}) or feat_shape: {feat_shape} != ({D, D, D, C_feat})")
                continue

            data_files.append(mat_fp)
            feat_files.append(feat_fp)
            mask_files.append(mask_fp)
            obj_ids.append(obj_id)
        return data_files, feat_files, mask_files, obj_ids

    # ---------- scaling utils ------------------------------------------
    def _scale(self, x, lo, hi):
        """min‑max to [‑1,1] (expects x already log‑scaled where needed)."""
        x = np.clip(x, lo, hi)
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    # def _create_and_save_mask(self, mat_fp, mask_fp):
    #     """Create mask based on material_id != background_id and save it."""
    #     mat = np.load(mat_fp)
    #     background_id = self.cfg.training.background_id
    #     mask = (mat[..., -1] != background_id).astype(np.float32)  # 1 for foreground, 0 for background
    #     np.save(mask_fp, mask)
    #     return mask

    def _load_clip_features_mask(self, mask_fp):
        """Load the pre-computed clip_features_mask.npy file."""
        if not os.path.exists(mask_fp):
            raise FileNotFoundError(f"clip_features_mask.npy not found at {mask_fp}. Please run voxelization first.")
        mask = np.load(mask_fp).astype(np.float32)
        return mask

    # ---------- torch Dataset interface --------------------------------
    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, idx):
        # --- load material grid (D,H,W,material_channels) --------------------------------
        mat = np.load(self.data_files[idx]).astype(np.float32)
        # --- load CLIP features (D,H,W,feature_channels) -------------------------------
        feat = np.load(self.feature_files[idx]).astype(np.float32)
        
        if self.feature_type == "occupancy" and feat.ndim == 3:
            feat = np.expand_dims(feat, axis=-1) # (D,D,D) -> (D,D,D,1)
        
        # --- load clip features mask -----------------------------------------
        mask_fp = self.mask_files[idx]
        # if os.path.exists(mask_fp):
        #     mask = np.load(mask_fp).astype(np.float32)
        # else:
        #     mask = self._create_and_save_mask(self.data_files[idx], mask_fp)
        mask = self._load_clip_features_mask(mask_fp)
        
        # --- extract material properties and IDs ---
        density = mat[..., 0]
        E = mat[..., 1]
        nu = mat[..., 2]
        mat_id_np = mat[..., -1]  # Last channel is material_id
        
        # --- safeguard: assert mask consistency with material_id background ---
        background_id = self.cfg.training.background_id
        expected_mask = (mat_id_np != background_id).astype(np.float32)
        
        if self.cfg.training.enforce_mask_consistency:
            assert np.array_equal(mask, expected_mask), \
                f"Mask inconsistency for {self.obj_ids[idx]}: clip_features_mask.npy doesn't match material_id-based mask"
        
        # --- create info dictionary with object ID ------------------------
        info = {
            "obj_id": self.obj_ids[idx], 
            "sample_id": self.cfg.training.sample_id,
            "data_path": self.data_files[idx],
            "feature_path": self.feature_files[idx],
            "mask_path": mask_fp
        }
        
        if self.to_normalize:
            # mat = mat.transpose(3, 0, 1, 2)
            # feat = feat.transpose(3, 0, 1, 2)
            # return mat, feat, mask, info

            # continuous channels ------------------------------------------------
            density = np.log10(density + 1e-6)  # log scale
            E       = np.log10(E + 1e-6)

            density = self._scale(
                density, self.cfg.training.density_min, self.cfg.training.density_max
            )
            E = self._scale(E, self.cfg.training.E_min, self.cfg.training.E_max)
            nu = self._scale(nu, self.cfg.training.nu_min, self.cfg.training.nu_max)

        # categorical channel ------------------------------------------------
        mat_id = torch.from_numpy(mat_id_np.astype(np.int64))  # Last channel is material_id

        # concat –> (D,H,W, 3+K)
        mat_cont   = np.stack([density, E, nu], axis=-1)        # (D,H,W,3)
        mat_cont = torch.from_numpy(mat_cont).permute(3, 0, 1, 2)
        feat = torch.from_numpy(feat).permute(3, 0, 1, 2)  # (feature_channels,D,H,W)
        mask = torch.from_numpy(mask)  # (D,H,W)

        return feat, mat_cont, mat_id, mask, info


# ----------------------------------------------------------------------------
# Dataset for segmentation
# ----------------------------------------------------------------------------
class MaterialSegmentationDataset(MaterialVoxelDataset):
    """Load CLIP feature grids and corresponding *discrete* material_id grids.

    Each sample directory must contain
        ├─ clip_features_features.npy      (D,D,D,feature_channels) float32
        └─ sample_<sample_id>/material_grid.npy (D,D,D,material_channels) float32 where the
           last channel is the integer material id (0‑background_id-1).
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        feat, mat_cont, mat_id, mask, info = super().__getitem__(idx)
        return feat, mat_id, mask, info

class MaterialVoxelDatasetContinuous(MaterialVoxelDataset):
    """
    Extends MaterialVoxelDataset but only returns continuous values (density, E, nu).
    Each sample directory must contain
        ├─ clip_features_features.npy      (grid_size,grid_size,grid_size,feature_channels) float32
        ├─ clip_features_mask.npy          (grid_size,grid_size,grid_size) float32
        └─ sample_0
           └─ material_grid.npy            (grid_size,grid_size,grid_size,material_channels)   float32
           # └─ mask.npy                     (grid_size,grid_size,grid_size)     float32 (optional) - OLD REFERENCE
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        # feat_torch, mat_continuous_torch, mat_id_torch, mask_torch, info
        feat, mat_cont, mat_id, mask, info = super().__getitem__(idx)
        return feat, mat_cont, mask, info

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    cfg = load_normalization_ranges(cfg)
    combined_dataset = MaterialVoxelDataset(cfg)
    # discrete_dataset = MaterialSegmentationDataset(cfg)
    # continuous_dataset = MaterialVoxelDatasetContinuous(cfg)
    # print("len(discrete_dataset): ", len(discrete_dataset))
    # print("len(continuous_dataset): ", len(continuous_dataset))
    if cfg.feature_type == "clip":
        D = cfg.features.clip.feature_channels
    elif cfg.feature_type == "rgb":
        D = cfg.features.rgb.feature_channels
    elif cfg.feature_type == "occupancy":
        D = cfg.features.occupancy.feature_channels
    else:
        raise ValueError(f"Invalid feature type: {cfg.feature_type}")

    print(f"len(combined_dataset): {len(combined_dataset)}")
    filtered_indices = [i for i, sample_obj_id in enumerate(combined_dataset.obj_ids) if sample_obj_id == "pixie"]
    print(f"len(filtered_indices): {len(filtered_indices)}")
    # for i in tqdm(range(len(discrete_dataset))):
    # # for i in tqdm(range(10)):
    #     feat, mat_id, mask, info = discrete_dataset[i]
    #     feat_cont, mat_cont, mask_cont, info_cont = continuous_dataset[i]
    #     feat_combined, mat_cont_combined, mat_id_combined, mask_combined, info_combined = combined_dataset[i]
    #     assert feat.shape == feat_cont.shape == (D, 64, 64, 64), f"feat.shape: {feat.shape}, feat_cont.shape: {feat_cont.shape}"
    #     # assert mat_id.shape == mat_id_cont.shape, f"mat_id.shape: {mat_id.shape}, mat_id_cont.shape: {mat_id_cont.shape}"
    #     assert mask.shape == mask_cont.shape, f"mask.shape: {mask.shape}, mask_cont.shape: {mask_cont.shape}"
    #     assert mat_id.shape == (64, 64, 64), f"mat_id.shape: {mat_id.shape}"
    #     assert mat_cont.shape == (3, 64, 64, 64), f"mat_cont.shape: {mat_cont.shape}"
    #     assert info == info_cont

    #     assert feat_combined.shape == feat.shape, f"feat_combined.shape: {feat_combined.shape}, feat.shape: {feat.shape}"
    #     assert mat_cont_combined.shape == mat_cont.shape, f"mat_cont_combined.shape: {mat_cont_combined.shape}, mat_cont.shape: {mat_cont.shape}"
    #     assert mat_id_combined.shape == mat_id.shape, f"mat_id_combined.shape: {mat_id_combined.shape}, mat_id.shape: {mat_id.shape}"
    #     assert mask_combined.shape == mask.shape, f"mask_combined.shape: {mask_combined.shape}, mask.shape: {mask.shape}"
    #     assert info_combined == info, f"info_combined: {info_combined}, info: {info}"



if __name__ == "__main__":
    main()