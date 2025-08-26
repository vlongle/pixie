#!/usr/bin/env python3
"""
$ python collect_stats.py --cfg ../trainer/conf/discrete_config.yaml \
                          --out stats_material_voxels.npz
"""
import argparse, json, numpy as np
from tqdm import tqdm
from collections import Counter
from pathlib import Path

from my_data import load_config, MaterialVoxelDataset, BACKGROUND_ID

# ---------------------------------------------------------------------
# running mean / variance  (Welford 1-pass, memory-safe)
# ---------------------------------------------------------------------
class RunningMoments:
    def __init__(self): self.n=0; self.mu=0.0; self.M2=0.0
    def update(self, arr):
        arr = arr.ravel()
        self.n += arr.size
        delta = arr - self.mu
        self.mu += delta.sum() / self.n
        self.M2 += ((arr - self.mu) * delta).sum()
    def stats(self):
        var = self.M2 / max(self.n-1,1)
        return dict(count=int(self.n), mean=float(self.mu), std=float(np.sqrt(var)))

# ---------------------------------------------------------------------
def unscale(x_scaled, lo, hi):
    """Invert min-max scaling from [-1,1] back to raw (log10) value."""
    return (x_scaled + 1.0) * (hi - lo) / 2.0 + lo

def main(cfg_yaml: str, out_file: str, nbins:int=100):

    # ────────────────────────────────────────────────────────────
    # 0.  load Hydra cfg & dataset
    # ────────────────────────────────────────────────────────────
    cfg = load_config(config_path=Path(cfg_yaml).parent,
                      config_name=Path(cfg_yaml).stem) if cfg_yaml else load_config()
    ds  = MaterialVoxelDataset(cfg)

    # histogram ranges (use cfg’s training mins/maxs)
    h_range = dict(
        logrho=(cfg.density_min, cfg.density_max),   # still log10
        logE   =(cfg.E_min,       cfg.E_max),        # still log10
        nu     =(cfg.nu_min,      cfg.nu_max),       # raw ν
    )
    bins = {k: np.linspace(*h_range[k], nbins+1) for k in h_range}

    h_logrho = np.zeros(nbins, np.int64)
    h_logE   = np.zeros(nbins, np.int64)
    h_nu     = np.zeros(nbins, np.int64)
    mat_cnt  = Counter()

    rm_logrho = RunningMoments(); rm_logE = RunningMoments(); rm_nu = RunningMoments()

    # ────────────────────────────────────────────────────────────
    for feat, mat_cont, mat_id, mask, _ in tqdm(ds, desc="collect"):
        m = mask.bool().numpy()

        # un-scale (broadcast-safe tensors → numpy)
        dens_log = unscale(mat_cont[0].numpy(), cfg.density_min, cfg.density_max)  # log10 ρ
        E_log    = unscale(mat_cont[1].numpy(), cfg.E_min,       cfg.E_max)        # log10 E
        nu_raw   = unscale(mat_cont[2].numpy(), cfg.nu_min,      cfg.nu_max)       # ν

        # select foreground voxels
        dens_fg = dens_log[m];  E_fg = E_log[m];  nu_fg = nu_raw[m]
        ids_fg  = mat_id.numpy()[m]

        rm_logrho.update(dens_fg); rm_logE.update(E_fg); rm_nu.update(nu_fg)
        h_logrho += np.histogram(dens_fg, bins=bins["logrho"])[0]
        h_logE   += np.histogram(E_fg,    bins=bins["logE"])[0]
        h_nu     += np.histogram(nu_fg,   bins=bins["nu"])[0]
        mat_cnt.update(ids_fg.tolist())

    # ────────────────────────────────────────────────────────────
    # save
    np.savez_compressed(
        out_file,
        hdr=json.dumps({"nbins":nbins,"hist_range":h_range}),
        h_density = h_logrho,      # keep key names identical for plot_stats.py
        h_logE    = h_logE,
        h_nu      = h_nu,
        density_stats=json.dumps(rm_logrho.stats()),
        logE_stats   =json.dumps(rm_logE.stats()),
        nu_stats     =json.dumps(rm_nu.stats()),
        mat_ids   =np.array(list(mat_cnt.keys()), dtype=np.int32),
        mat_freqs =np.array(list(mat_cnt.values()), dtype=np.int64),
    )
    print(f"[done] stats (foreground voxels only) → {out_file}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--cfg", help="path to Hydra .yaml (optional)")
    pa.add_argument("--out", default="stats_material_voxels.npz")
    pa.add_argument("--nbins", type=int, default=100)
    args = pa.parse_args()
    main(args.cfg, args.out, args.nbins)
