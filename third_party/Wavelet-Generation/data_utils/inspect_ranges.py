#!/usr/bin/env python3
import os
import numpy as np
import glob
import logging
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from rich.console import Console
from rich.table import Table
import shutil
import hydra
from omegaconf import DictConfig
from pixie.utils import set_logger
# Add the parent directory to sys.path to import pixie utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pixie.utils import resolve_paths, validate_config, load_json, save_json

def process_file(fp, cfg):
    """Process a single material grid file and extract statistics."""
    obj_id = fp.split("/")[-3]
    feat_fp = f"{cfg.paths.render_outputs_dir}/{obj_id}/clip_features_features.npy"
    grid_size = cfg.training.default_grid_size
    background_id = cfg.training.background_id
    
    ok = (
        os.path.exists(feat_fp)
        and os.path.exists(fp)
    )
    if not ok:
        return {'status': 'error', 'obj_id': obj_id, 'reason': 'missing files'}

    try:
        mat_shape = np.load(fp, mmap_mode='r').shape
        feat_shape = np.load(feat_fp, mmap_mode='r').shape
        if not (
            mat_shape == (grid_size, grid_size, grid_size, cfg.training.in_material_channels)
            and feat_shape == (grid_size, grid_size, grid_size, cfg.training.normalization.clip_feature_channels)
        ):
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'invalid shape'}

        mat = np.load(fp, mmap_mode="r")
        mask = mat[..., 3] != background_id
        if not mask.any():
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'no foreground'}

        # Get material statistics
        material_ids = mat[..., 3].astype(int)
        material_counts = Counter(material_ids.flatten())
        total_voxels = material_ids.size

        # Get physical properties
        dens = mat[..., 0][mask]
        E = mat[..., 1][mask]
        nu = mat[..., 2][mask]

        # Validate data to prevent NaN issues
        if np.any(dens < 0):
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'negative density'}
        if np.any(E < 0):
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'negative Young\'s modulus'}
        if np.any(np.isnan(dens)) or np.any(np.isnan(E)) or np.any(np.isnan(nu)):
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'NaN values'}
        if np.any(np.isinf(dens)) or np.any(np.isinf(E)) or np.any(np.isinf(nu)):
            return {'status': 'error', 'obj_id': obj_id, 'reason': 'infinite values'}

        # Reservoir sampling for each property
        def reservoir_sample(arr, cap):
            if len(arr) <= cap:
                return arr
            indices = np.random.choice(len(arr), cap, replace=False)
            return arr[indices]

        reservoir_cap = cfg.training.normalization.reservoir_cap
        return {
            'status': 'success',
            'material_counts': material_counts,
            'total_voxels': total_voxels,
            'dens': reservoir_sample(dens, reservoir_cap),
            'E': reservoir_sample(E, reservoir_cap),
            'nu': reservoir_sample(nu, reservoir_cap),
            'min_max': {
                'd': (dens.min(), dens.max()),
                'E': (E.min(), E.max()),
                'nu': (nu.min(), nu.max())
            }
        }
    except Exception as e:
        return {'status': 'error', 'obj_id': obj_id, 'reason': f'exception: {str(e)}'}

def save_statistics(stats, cfg):
    """Save computed statistics to files."""
    # Create output directory
    output_dir = Path(cfg.paths.normalization_stats_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed statistics as JSON
    stats_file = output_dir / "material_statistics.json"
    save_json(stats, stats_file)
    logging.info(f"Saved detailed statistics to {stats_file}")
    
    # Save normalization ranges for easy loading
    normalization_ranges = {
        'density_min': float(stats['normalization_ranges']['density_p1']),
        'density_max': float(stats['normalization_ranges']['density_p99']),
        'E_min': float(stats['normalization_ranges']['E_p1']),
        'E_max': float(stats['normalization_ranges']['E_p99']),
        'nu_min': float(stats['normalization_ranges']['nu_p1']),
        'nu_max': float(stats['normalization_ranges']['nu_p99']),
        'density_p1': float(stats['normalization_ranges']['density_p1']),
        'density_p99': float(stats['normalization_ranges']['density_p99']),
        'E_p1': float(stats['normalization_ranges']['E_p1']),
        'E_p99': float(stats['normalization_ranges']['E_p99']),
        'nu_p1': float(stats['normalization_ranges']['nu_p1']),
        'nu_p99': float(stats['normalization_ranges']['nu_p99']),
    }
    
    ranges_file = output_dir / "normalization_ranges.yaml"
    save_json(normalization_ranges, ranges_file)
    
    logging.info(f"Saved normalization ranges to {ranges_file}")
    
    return stats_file, ranges_file


def compute_dataset_statistics(cfg):
    """Compute statistics across the entire dataset."""
    paths = glob.glob(f"{cfg.paths.render_outputs_dir}/*/sample_0/material_grid.npy")
    
    if not paths:
        raise ValueError(f"No material grid files found in {cfg.paths.render_outputs_dir}")
    
    logging.info(f"Found {len(paths)} material grid files to process")
    
    # Process files in parallel
    n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    process_func = partial(process_file, cfg=cfg)
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, paths),
            total=len(paths),
            desc="Processing files"
        ))
    
    # Separate successful and failed results
    successful_results = []
    failed_results = []
    for result in results:
        if result['status'] == 'success':
            successful_results.append(result)
        else:
            failed_results.append(result)

    # Save problematic object IDs and detailed reasons
    output_dir = Path(cfg.paths.normalization_stats_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if failed_results:
        # Save problematic object IDs as JSON
        problematic_file = output_dir / 'problematic_objects.json'
        problematic_ids = [result['obj_id'] for result in failed_results]
        save_json(problematic_ids, problematic_file)
        
        # Save detailed information about problematic objects
        detailed_problematic_file = output_dir / 'problematic_objects_detailed.json'
        save_json(failed_results, detailed_problematic_file)
        
        # Group by reason for summary
        reason_counts = Counter(result.get('reason', 'unknown') for result in failed_results)
        
        logging.info(f"Found {len(failed_results)} problematic objects:")
        for reason, count in sorted(reason_counts.items()):
            logging.info(f"  - {reason}: {count} objects")
        logging.info(f"Problematic object IDs saved to: {problematic_file}")
        logging.info(f"Detailed info saved to: {detailed_problematic_file}")

    logging.info(f"Total valid objects: {len(successful_results)}")

    # Aggregate results
    material_counts = Counter()
    total_voxels = 0
    reservoir_d = []
    reservoir_E = []
    reservoir_nu = []
    d_min = d_max = E_min = E_max = nu_min = nu_max = None

    for result in tqdm(successful_results, desc="Aggregating results"):
        material_counts.update(result['material_counts'])
        total_voxels += result['total_voxels']
        
        # Update min/max
        d_min = result['min_max']['d'][0] if d_min is None else min(d_min, result['min_max']['d'][0])
        d_max = result['min_max']['d'][1] if d_max is None else max(d_max, result['min_max']['d'][1])
        E_min = result['min_max']['E'][0] if E_min is None else min(E_min, result['min_max']['E'][0])
        E_max = result['min_max']['E'][1] if E_max is None else max(E_max, result['min_max']['E'][1])
        nu_min = result['min_max']['nu'][0] if nu_min is None else min(nu_min, result['min_max']['nu'][0])
        nu_max = result['min_max']['nu'][1] if nu_max is None else max(nu_max, result['min_max']['nu'][1])
        
        # Combine reservoir samples
        reservoir_d.extend(result['dens'])
        reservoir_E.extend(result['E'])
        reservoir_nu.extend(result['nu'])

    # Final reservoir sampling if needed
    reservoir_cap = cfg.training.normalization.reservoir_cap
    if len(reservoir_d) > reservoir_cap:
        indices = np.random.choice(len(reservoir_d), reservoir_cap, replace=False)
        reservoir_d = np.array(reservoir_d)[indices]
        reservoir_E = np.array(reservoir_E)[indices]
        reservoir_nu = np.array(reservoir_nu)[indices]

    # ------------------- compute statistics -------------------------------------------
    def pct(arr, p):
        """percentile if array not empty, else nan"""
        arr = np.asarray(arr)
        return np.percentile(arr, p) if arr.size else float("nan")

    # Log transform for density and Young's modulus
    dens_log = np.log10(np.maximum(np.asarray(reservoir_d), 1e-6))
    E_log = np.log10(np.maximum(np.asarray(reservoir_E), 1e-6))
    nu_arr = np.asarray(reservoir_nu)

    # Compute comprehensive statistics
    stats = {
        'dataset_info': {
            'total_objects': len(successful_results),
            'failed_objects': len(failed_results),
            'total_voxels': int(total_voxels),
        },
        'material_distribution': {
            str(mat_id): {
                'count': int(count),
                'percentage': float((count / total_voxels) * 100)
            }
            for mat_id, count in sorted(material_counts.items())
        },
        'raw_ranges': {
            'density': {'min': float(d_min), 'max': float(d_max)},
            'youngs_modulus': {'min': float(E_min), 'max': float(E_max)},
            'poisson_ratio': {'min': float(nu_min), 'max': float(nu_max)},
        },
        'log_ranges': {
            'density': {
                'min': float(dens_log.min()),
                'max': float(dens_log.max()),
                'p1': float(pct(dens_log, 1)),
                'p99': float(pct(dens_log, 99))
            },
            'youngs_modulus': {
                'min': float(E_log.min()),
                'max': float(E_log.max()),
                'p1': float(pct(E_log, 1)),
                'p99': float(pct(E_log, 99))
            }
        },
        'poisson_percentiles': {
            f'p{p:02d}': float(pct(nu_arr, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        'normalization_ranges': {
            'density_p1': float(pct(dens_log, 1)),
            'density_p99': float(pct(dens_log, 99)),
            'E_p1': float(pct(E_log, 1)),
            'E_p99': float(pct(E_log, 99)),
            'nu_p1': float(pct(nu_arr, 1)),
            'nu_p99': float(pct(nu_arr, 99)),
        }
    }

    # Print summary report using rich
    console = Console()
    
    # --- Material ID Distribution ---
    table = Table(title="Material ID Distribution")
    table.add_column("Material ID", justify="right", style="cyan")
    table.add_column("Voxel Count", justify="right", style="magenta")
    table.add_column("Percentage", justify="right", style="green")
    
    for mat_id, count in sorted(material_counts.items()):
        percentage = (count / total_voxels) * 100
        table.add_row(str(mat_id), f"{count:,}", f"{percentage:.2f}%")
    
    console.print(table)
    
    # --- Main Statistics Table ---
    table = Table(title="Physical Property Statistics")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Min", justify="right", style="magenta")
    table.add_column("Max", justify="right", style="magenta")
    table.add_column("Min (log10)", justify="right", style="green")
    table.add_column("Max (log10)", justify="right", style="green")
    table.add_column("p1", justify="right", style="yellow")
    table.add_column("p99", justify="right", style="red")

    table.add_row(
        "Density",
        f"{d_min:.4g}", f"{d_max:.4g}",
        f"{dens_log.min():.3f}", f"{dens_log.max():.3f}",
        f"{pct(dens_log, 1):.3f}", f"{pct(dens_log, 99):.3f}"
    )
    table.add_row(
        "Young's E",
        f"{E_min:.4g}", f"{E_max:.4g}",
        f"{E_log.min():.3f}", f"{E_log.max():.3f}",
        f"{pct(E_log, 1):.3f}", f"{pct(E_log, 99):.3f}"
    )
    table.add_row(
        "Poisson ν",
        f"{nu_min:.4f}", f"{nu_max:.4f}",
        "_", "_",  # Log scale is not used for Poisson's ratio
        f"{pct(nu_arr, 1):.4f}", f"{pct(nu_arr, 99):.4f}"
    )
    console.print(table)

    return stats


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to compute material property statistics across the dataset."""
    set_logger()
    
    validate_config(cfg, single_obj=False)
    cfg = resolve_paths(cfg)
    
    # Clean up old normalization stats before recomputing
    output_dir = Path(cfg.paths.normalization_stats_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    
    logging.info("Computing dataset statistics for material properties...")
    
    stats = compute_dataset_statistics(cfg)
    stats_file, ranges_file = save_statistics(stats, cfg)
    
    console = Console()
    console.print(f"\n[bold green]✓ Statistics computation completed![/bold green]")
    console.print(f"Detailed statistics saved to: [cyan]{stats_file}[/cyan]")
    console.print(f"Normalization ranges saved to: [cyan]{ranges_file}[/cyan]")
    console.print(f"You can use the ranges from [cyan]{ranges_file}[/cyan] to update your training config.")


if __name__ == "__main__":
    main()
