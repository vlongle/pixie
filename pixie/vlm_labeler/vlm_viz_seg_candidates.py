import argparse
import os
import glob
import json
from pixie.utils import str2bool
from vlmx.utils import load_json
from pixie.utils import run_cmd
import logging

def run_vlm_viz_seg_candidates(obj_id: str, vlm_seg_results_dir: str, 
                              render_outputs_dir: str, outputs_dir: str,
                              grid_size: int = 200, gray_threshold: float = 0.05,
                              overwrite: bool = False):
    """Run VLM visualization of segmentation candidates."""
    
    # Load VLM segmentation results
    vlm_results_path = os.path.join(vlm_seg_results_dir, obj_id, "vlm_results.json")
    if not os.path.exists(vlm_results_path):
        logging.info(f"VLM results not found at {vlm_results_path}")
        return
    
    result = load_json(vlm_results_path)

    # Check if visualization already exists
    output_dir = os.path.join(vlm_seg_results_dir, obj_id, "query_0")
    if not overwrite and os.path.exists(os.path.join(output_dir, "clip.png")):
        logging.info(f"Skipping {obj_id} because {output_dir}/clip.png exists")
        return
    
    # Generate voxel features if needed
    grid_size = grid_size
    voxel_size = 1.0 / grid_size
    method = "f3rm"
    render_output_dir = os.path.join(render_outputs_dir, obj_id)
    
    # Find the latest config file in the output directory
    output_dir = os.path.join(outputs_dir, obj_id, method)
    if not os.path.exists(output_dir):
        logging.info(f"Output directory not found: {output_dir}")
        return
        
    latest_run = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
    config_path = os.path.join(latest_run, "config.yml")

    should_run_voxel = (
        overwrite or
        not (os.path.exists(f"{render_output_dir}/clip_features.npz") and
             os.path.exists(f"{render_output_dir}/clip_features_pc.ply")))
    
    if should_run_voxel:
        # Generate voxel features
        voxel_cmd = [
            "python", "pixie/voxel/voxelize.py",
            "--scene", config_path,
            "--output", f"{render_output_dir}/clip_features.npz",
            "--voxel_size", str(voxel_size),
            "--gray_threshold", str(gray_threshold)
        ]
        run_cmd(voxel_cmd, step_name="VOXEL_TO_PC")

    # Generate visualizations for each query
    for i, query in enumerate(result["all_queries"]):
        query_str = ", ".join(query)
        output_dir = os.path.join(vlm_seg_results_dir, obj_id, f"query_{i}")
        
        # Use string command with proper quoting
        viz_cmd = (f'python pixie/voxel/viz_segmentation.py '
                  f'--obj_id {obj_id} '
                  f'--output_dir "{output_dir}" '
                  f'--part_queries "{query_str}" '
                  f'--render_outputs_dir "{render_output_dir}" '
                  f'--overwrite {overwrite}')
        run_cmd(viz_cmd, step_name=f"VIZ_SEG_QUERY_{i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--vlm_seg_results_dir", type=str, required=True)
    parser.add_argument("--render_outputs_dir", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument("--grid_size", type=int, default=200)
    parser.add_argument("--gray_threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    run_vlm_viz_seg_candidates(
        obj_id=args.obj_id,
        vlm_seg_results_dir=args.vlm_seg_results_dir,
        render_outputs_dir=args.render_outputs_dir,
        outputs_dir=args.outputs_dir,
        grid_size=args.grid_size,
        gray_threshold=args.gray_threshold,
        overwrite=args.overwrite
    )
