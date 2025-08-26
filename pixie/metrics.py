import os
import numpy as np
from typing import List, Optional
from omegaconf import DictConfig
from .utils import load_json
from typing import Dict
from collections import defaultdict
import math
from .utils import save_json, get_obj_class_for_id
import pandas as pd



def get_obj_ids_for_class(obj_class: str, cfg: DictConfig, only_valid: bool = True) -> List[str]:
    """Get all object IDs for a given object class.
    
    Args:
        obj_class: The object class to search for
        cfg: Configuration containing metadata path
        only_valid: If True, only return objects where vlm_filtering.is_appropriate is True
    """
    if cfg and cfg.paths.obj_metadata_path:
        # Handle relative paths by making them absolute relative to the project root
        metadata_path = cfg.paths.obj_metadata_path
        obj_metadata = load_json(metadata_path)
        obj_ids = []
        
        # Iterate through all objects and find those matching the class
        for obj_id, metadata in obj_metadata.items():
            if metadata.get("obj_class", "UNKNOWN") == obj_class:
                # If only_valid is True, check vlm_filtering.is_appropriate
                if only_valid:
                    vlm_filtering = metadata.get("vlm_filtering", {})
                    if vlm_filtering.get("is_appropriate", False):
                        obj_ids.append(obj_id)
                else:
                    obj_ids.append(obj_id)
        
        return obj_ids
    
    return []


def aggregate_material_statistics(cfg: DictConfig, obj_ids: List[str]) -> dict:
    """Aggregate material statistics from VLM results across objects."""
    stats = {'E': [], 'density': [], 'nu': [], 'parts': [], 'material_ids': [], 'obj_ids': []}
    
    for obj_id in obj_ids:
        base_dir = os.path.join(cfg.paths.vlm_seg_mat_sample_results_dir, obj_id)
        sample_id = f"sample_{cfg.physics.sample_id}"
        chosen_path = os.path.join(base_dir, sample_id, "chosen_vlm_results.json")
        
        if os.path.exists(chosen_path):
            
            vlm_results = load_json(chosen_path)
            material_dict = vlm_results['material_dict']
            
            for part_name, props in material_dict.items():
                for key in ['E', 'density', 'nu']:
                    if key in props:
                        stats[key].append(props[key])
                
                if 'material_id' in props:
                    stats['material_ids'].append(props['material_id'])
                
                stats['parts'].append(part_name)
                stats['obj_ids'].append(obj_id)
    
    return stats


def remove_outliers_percentile(data: np.ndarray, lower: float = 5, upper: float = 95) -> np.ndarray:
    """Remove outliers using percentile bounds."""
    p_low, p_high = np.percentile(data, [lower, upper])
    return data[(data >= p_low) & (data <= p_high)]

def clean_categorical(data: np.ndarray, min_freq: float = 0.1) -> np.ndarray:
    """Remove rare categorical values below min_freq percentage of total data.
    default is 1%"""
    unique, counts = np.unique(data, return_counts=True)
    min_count = int(len(data) * min_freq)
    valid_categories = unique[counts >= min_count]
    return data[np.isin(data, valid_categories)]


def clean_continuous(data: np.ndarray, prop_type: str = 'E') -> np.ndarray:
    """Clean material property data with physical constraints + percentile filtering."""
    # Apply physical constraints first
    if prop_type == 'E':  # Young's modulus
        data = data[data > 0]  # Must be positive
    elif prop_type == 'density':
        data = data[data > 0]  # Must be positive  
    elif prop_type == 'nu':  # Poisson's ratio
        data = data[(data >= -1) & (data <= 0.5)]  # Physical bounds
    
    # Then apply percentile filtering to remove extreme outliers
    return remove_outliers_percentile(data, lower=2.5, upper=97.5)


# ============================================================================
# Inference Metrics Functions
# ============================================================================


class InferenceMetrics:
    """Container for tracking inference metrics."""
    def __init__(self):
        self.seg_accuracies = []
        self.cont_mse_values = []
        self.density_mse_values = []
        self.youngs_mse_values = []
        self.poisson_mse_values = []
        self.obj_metrics = {}
        self.local_obj_ids = []
    
    def add_batch_metrics(self, seg_acc, cont_mse, density_mse, youngs_mse, poisson_mse):
        """Add batch-level metrics."""
        self.seg_accuracies.append(seg_acc)
        self.cont_mse_values.append(cont_mse)
        self.density_mse_values.append(density_mse)
        self.youngs_mse_values.append(youngs_mse)
        self.poisson_mse_values.append(poisson_mse)
    
    def add_sample_metrics(self, obj_id, metrics_dict):
        """Add sample-level metrics for an object."""
        if obj_id not in self.obj_metrics:
            self.obj_metrics[obj_id] = defaultdict(list)
        
        for key, value in metrics_dict.items():
            self.obj_metrics[obj_id][key].append(value)
    
    def gather_all_metrics(self, rank, world_size):
        """Gather metrics from all processes."""
        if world_size == 1:
            return [self.seg_accuracies], [self.cont_mse_values], [self.density_mse_values], \
                   [self.youngs_mse_values], [self.poisson_mse_values], [self.local_obj_ids], \
                   [self.obj_metrics]
        
        import torch.distributed as dist
        all_metrics = [None for _ in range(world_size)]
        metrics_to_gather = [
            self.seg_accuracies, self.cont_mse_values, self.density_mse_values,
            self.youngs_mse_values, self.poisson_mse_values, self.local_obj_ids,
            self.obj_metrics
        ]
        
        gathered = []
        for metric in metrics_to_gather:
            all_values = [None for _ in range(world_size)]
            dist.gather_object(metric, all_values if rank == 0 else None, dst=0)
            gathered.append(all_values)
        
        return gathered


def save_metrics_file(output_dir, seg_ckpt, cont_ckpt, global_avgs, obj_avgs, 
                     merged_metrics, ci_low, ci_high, se, z, disp_label):
    """Save detailed metrics to JSON file."""
    # Calculate dispersions for all metrics
    dispersions = {}
    for metric in ["cont_mse", "density_mse", "youngs_mse", "poisson_mse"]:
        values = [avg[metric] for avg in obj_avgs.values()]
        if len(values) > 1:
            if disp_label == "SE":
                dispersions[metric] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            else:
                dispersions[metric] = float(np.std(values, ddof=0))
        else:
            dispersions[metric] = 0.0
    
    # Structure all metrics data
    metrics_data = {
        "checkpoints": {
            "segmentation": seg_ckpt,
            "continuous": cont_ckpt
        },
        "summary": {
            "total_objects": len(obj_avgs),
            "dispersion_type": disp_label,
            "segmentation_accuracy": {
                "mean": float(global_avgs['seg_acc']),
                "confidence_interval_90": {
                    "low": float(ci_low),
                    "high": float(ci_high),
                    "margin": float(z * se)
                }
            },
            "continuous_metrics": {
                "overall": {
                    "mean": float(global_avgs['cont_mse']),
                    "dispersion": dispersions['cont_mse']
                },
                "density": {
                    "mean": float(global_avgs['density_mse']),
                    "dispersion": dispersions['density_mse']
                },
                "youngs_modulus": {
                    "mean": float(global_avgs['youngs_mse']),
                    "dispersion": dispersions['youngs_mse']
                },
                "poisson_ratio": {
                    "mean": float(global_avgs['poisson_mse']),
                    "dispersion": dispersions['poisson_mse']
                }
            }
        },
        "per_object_metrics": obj_avgs,
        "per_object_raw_data": {
            oid: {
                "segmentation_accuracy": {
                    "values": metrics["seg_acc"],
                    "mean": sum(metrics["seg_acc"]) / len(metrics["seg_acc"]) if metrics["seg_acc"] else 0.0
                },
                "continuous_mse": {
                    "values": metrics["cont_mse"],
                    "mean": sum(metrics["cont_mse"]) / len(metrics["cont_mse"]) if metrics["cont_mse"] else 0.0
                }
            }
            for oid, metrics in merged_metrics.items()
        }
    }
    
    save_json(metrics_data, os.path.join(output_dir, "metrics.json"))


def generate_class_table(cfg, obj_averages, use_sem, disp_label, output_dir):
    """Generate per-class metrics table and save as JSON."""
    class_buckets = defaultdict(list)
    for oid, metrics in obj_averages.items():
        cls_name = get_obj_class_for_id(oid, cfg)
        class_buckets[cls_name].append(metrics)
    
    # Structure the per-class data
    class_metrics = {}
    metric_keys = ["seg_acc", "cont_mse", "density_mse", "youngs_mse", "poisson_mse"]
    
    for cls, obj_list in class_buckets.items():
        class_data = {
            "n_objects": len(obj_list),
            "metrics": {}
        }
        
        for key in metric_keys:
            values = [obj[key] for obj in obj_list if key in obj]
            if not values:
                class_data["metrics"][key] = {
                    "mean": 0.0,
                    "dispersion": 0.0,
                    "dispersion_type": "CI_90" if key == "seg_acc" else disp_label
                }
                continue
                
            mean = float(np.mean(values))
            
            if key == "seg_acc":
                # 90% CI for proportion
                n = len(values)
                ci = float(1.645 * np.sqrt(mean * (1 - mean) / n)) if n > 0 else 0.0
                class_data["metrics"][key] = {
                    "mean": mean,
                    "dispersion": ci,
                    "dispersion_type": "CI_90"
                }
            else:
                # Standard error or std
                if use_sem:
                    disp = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                else:
                    disp = float(np.std(values, ddof=0))
                class_data["metrics"][key] = {
                    "mean": mean,
                    "dispersion": disp,
                    "dispersion_type": disp_label
                }
        
        class_metrics[cls] = class_data
    
    # Add overall statistics
    overall_data = {
        "n_objects": len(obj_averages),
        "metrics": {}
    }
    
    for key in metric_keys:
        values = [obj[key] for obj in obj_averages.values() if key in obj]
        if values:
            mean = float(np.mean(values))
            if key == "seg_acc":
                n = len(values)
                ci = float(1.645 * np.sqrt(mean * (1 - mean) / n)) if n > 0 else 0.0
                overall_data["metrics"][key] = {
                    "mean": mean,
                    "dispersion": ci,
                    "dispersion_type": "CI_90"
                }
            else:
                if use_sem:
                    disp = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                else:
                    disp = float(np.std(values, ddof=0))
                overall_data["metrics"][key] = {
                    "mean": mean,
                    "dispersion": disp,
                    "dispersion_type": disp_label
                }
        else:
            overall_data["metrics"][key] = {
                "mean": 0.0,
                "dispersion": 0.0,
                "dispersion_type": "CI_90" if key == "seg_acc" else disp_label
            }
    
    class_metrics["ALL"] = overall_data
    
    # Save as JSON
    save_json(class_metrics, os.path.join(output_dir, "per_class_metrics.json"))
    
    # Still print table for console output
    rows = []
    for cls, data in class_metrics.items():
        row = {"obj_class": cls, "n_objects": data["n_objects"]}
        for key in metric_keys:
            m = data["metrics"][key]
            row[key] = f"{m['mean']:.4f} ± {m['dispersion']:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows).sort_values("obj_class")
    print(f"\nPer-class metric breakdown [Dispersion: {disp_label}]\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.to_string(index=False))


def generate_metrics_report(cfg, all_metrics, output_dir, seg_checkpoint_path, cont_checkpoint_path,
                          dispersion="sem", print_table=True):
    """Generate and save comprehensive metrics report."""
    # Unpack metrics
    (all_seg_acc, all_cont_mse, all_density_mse, all_youngs_mse, 
     all_poisson_mse, all_obj_ids, all_obj_metrics) = all_metrics
    
    # Flatten metrics
    flat_metrics = {}
    for name, metric_list in [
        ("seg_acc", all_seg_acc),
        ("cont_mse", all_cont_mse),
        ("density_mse", all_density_mse),
        ("youngs_mse", all_youngs_mse),
        ("poisson_mse", all_poisson_mse)
    ]:
        flat_metrics[name] = [item for sublist in metric_list if sublist for item in sublist]
    
    # Merge per-object metrics
    merged_obj_metrics = {}
    for obj_dict in all_obj_metrics:
        if obj_dict is None:
            continue
        for oid, metrics in obj_dict.items():
            if oid not in merged_obj_metrics:
                merged_obj_metrics[oid] = defaultdict(list)
            for key, values in metrics.items():
                merged_obj_metrics[oid][key].extend(values)
    
    # Calculate object-level averages
    obj_averages = {}
    for oid, metrics in merged_obj_metrics.items():
        obj_averages[oid] = {k: sum(v)/len(v) if v else 0.0 for k, v in metrics.items()}
    
    # Global averages
    global_avgs = {}
    for metric_name in ["seg_acc", "cont_mse", "density_mse", "youngs_mse", "poisson_mse"]:
        values = [obj_avg[metric_name] for obj_avg in obj_averages.values() 
                  if metric_name in obj_avg]
        global_avgs[metric_name] = np.mean(values) if values else 0.0
    
    # Calculate dispersions
    use_sem = dispersion.lower() in {"sem", "stderr"}
    
    def calc_dispersion(values):
        if not values or len(values) <= 1:
            return 0.0
        if use_sem:
            return np.std(values, ddof=1) / math.sqrt(len(values))
        return np.std(values, ddof=0)
    
    # Print results
    seg_se = calc_dispersion([avg["seg_acc"] for avg in obj_averages.values()])
    z_90 = 1.645
    seg_ci_low = global_avgs["seg_acc"] - z_90 * seg_se
    seg_ci_high = global_avgs["seg_acc"] + z_90 * seg_se
    
    disp_label = "SE" if use_sem else "STD"
    
    print(f"\nInference complete!")
    print(f"  Average Segmentation Accuracy: {global_avgs['seg_acc']:.4f}")
    print(f"    90% CI: [{seg_ci_low:.4f}, {seg_ci_high:.4f}] (± {z_90*seg_se:.4f})")
    print(f"  Average Continuous MSE: {global_avgs['cont_mse']:.6f} "
          f"({disp_label} {calc_dispersion([avg['cont_mse'] for avg in obj_averages.values()]):.6f})")
    
    for metric, label in [("density_mse", "Density"), ("youngs_mse", "Young's"), 
                         ("poisson_mse", "Poisson")]:
        disp = calc_dispersion([avg[metric] for avg in obj_averages.values()])
        print(f"    • {label} MSE: {global_avgs[metric]:.6f} ({disp_label} {disp:.6f})")
    
    # Save metrics file
    save_metrics_file(output_dir, seg_checkpoint_path, cont_checkpoint_path,
                     global_avgs, obj_averages, merged_obj_metrics, seg_ci_low, 
                     seg_ci_high, seg_se, z_90, disp_label)
    
    # Save object IDs
    flat_obj_ids = [item for sublist in all_obj_ids if sublist for item in sublist]
    unique_obj_ids = sorted(set(flat_obj_ids))
    save_json(unique_obj_ids, os.path.join(output_dir, "evaluated_obj_ids.json"))
    
    # Generate per-class table if requested
    if print_table:
        generate_class_table(cfg, obj_averages, use_sem, disp_label, output_dir)
