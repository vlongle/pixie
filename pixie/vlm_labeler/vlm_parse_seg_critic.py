import os
import argparse
import glob
import os.path as osp
from vlmx.utils import save_json, load_json
from vlmx.agent import Agent, AgentConfig
from tqdm import tqdm
import json
from PIL import Image
import logging
from vlmx.utils import seed_everything
import shutil
from pixie.utils import set_logger
import numpy as np


def run_vlm_parse_seg_critic(obj_id: str, vlm_seg_results_dir: str, 
                            vlm_seg_critic_results_dir: str, vlm_seg_mat_sample_results_dir: str,
                         model_name: str = "gemini-2.0-flash",
                            api_key: str = ""):
    """Run VLM parse segmentation critic to finalize results."""
    
    seed_everything(0)


    # Load critic results
    critic_chosen_path = os.path.join(vlm_seg_critic_results_dir, obj_id, "vlm_critic_results.json")
    if not os.path.exists(critic_chosen_path):
        logging.error(f"Critic results not found at {critic_chosen_path}")
        return
        
    critic_chosen_results = load_json(critic_chosen_path)
    
    # Determine best query
    try:
        best_query_id = int(critic_chosen_results["best_query"].split("_")[1])
    except:
        best_score = -1
        best_query_id = None
        for query_key, eval_data in critic_chosen_results["query_evaluations"].items():
            score = eval_data["score"]
            if score > best_score:
                best_score = score
                best_query_id = int(query_key.split("_")[1])
        if best_query_id is None:
            best_query_id = 0
    
    # Load VLM results
    json_path = os.path.join(vlm_seg_results_dir, obj_id, "vlm_results.json")
    if not os.path.exists(json_path):
        logging.error(f"VLM results not found at {json_path}")
        return
        
    old_json = load_json(json_path)
    best_query = old_json["all_queries"][best_query_id]
    logging.info(f">> BEST_QUERY: {best_query}")
    
    # Process material samples
    sample_mat_dir = os.path.join(vlm_seg_mat_sample_results_dir, obj_id)
    if not os.path.exists(sample_mat_dir):
        logging.error(f"Sample material directory not found: {sample_mat_dir}")
        return
        
    num_sample_mat = len([d for d in os.listdir(sample_mat_dir) if d.startswith("sample_")])
    logging.info(f">> NUM_SAMPLE_MAT: {num_sample_mat}")
    
    for i in range(num_sample_mat):
        # Make a copy of the json file
        new_json_path = os.path.join(sample_mat_dir, f"sample_{i}", "chosen_vlm_results.json")
        sample_json_path = os.path.join(sample_mat_dir, f"sample_{i}", "vlm_results.json")
        
        if not os.path.exists(sample_json_path):
            logging.warning(f"Sample JSON not found: {sample_json_path}")
            continue
            
        shutil.copy(sample_json_path, new_json_path)
        sample_json = load_json(sample_json_path)

        # Replace the keys in material_dict with the best query
        old_keys = list(sample_json["material_dict"].keys())
        new_material_dict = {}
        for old_key, new_key in zip(old_keys, best_query):
            new_material_dict[new_key] = sample_json["material_dict"][old_key]
        sample_json["material_dict"] = new_material_dict
        sample_json["all_queries"] = [best_query]
        save_json(sample_json, new_json_path)
        
        logging.info(f"Updated material sample {i} with best query")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--vlm_seg_results_dir", type=str, required=True)
    parser.add_argument("--vlm_seg_critic_results_dir", type=str, required=True)
    parser.add_argument("--vlm_seg_mat_sample_results_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    
    set_logger()
    run_vlm_parse_seg_critic(
        obj_id=args.obj_id,
        vlm_seg_results_dir=args.vlm_seg_results_dir,
        vlm_seg_critic_results_dir=args.vlm_seg_critic_results_dir,
        vlm_seg_mat_sample_results_dir=args.vlm_seg_mat_sample_results_dir,
        model_name=args.model_name,
        api_key=args.api_key
    )