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
from pixie.utils import str2bool, set_logger
import numpy as np
from vlm_seg_class_instruction import INSTRUCTION_FUNCTIONS


def run_vlm_segmentation(obj_id: str, obj_class: str, output_dir: str, 
                        data_dir: str, 
                        model_name: str, api_key: str, overwrite: bool = False,
                        num_alternative_queries: int = 2, input_num_views: int = 15):
    """Run VLM segmentation for a single object."""
    # Define the segmentation agent
    class SegmentationAgent(Agent):
        OUT_RESULT_PATH = "vlm_results.json"
        
        def _make_system_instruction(self):
            return INSTRUCTION_FUNCTIONS[obj_class](num_alternative_queries)

        def _make_prompt_parts(self, image_paths):
            images = [Image.open(image_path) for image_path in image_paths]
            question = ["The image is :"] + images
            return question

        def parse_response(self, response):
            json_str = response.text.strip().strip("```json").strip()
            parsed_response = json.loads(json_str, strict=False)
            print(parsed_response)
            save_json(parsed_response, os.path.join(
                self.cfg.out_dir, self.OUT_RESULT_PATH))
            return parsed_response

    # Create agent
    agent = SegmentationAgent(AgentConfig(
        model_name=model_name,
        out_dir=output_dir,
        api_key=api_key,
    ))

    # Get image paths
    all_image_paths = glob.glob(os.path.join(data_dir, obj_id, "train", "*.png"))
    if len(all_image_paths) < input_num_views:
        logging.warning(f"Not enough images found for {obj_id} (found {len(all_image_paths)}, need {input_num_views}). Skipping.")
        return

    # Randomly sample images
    random_image_paths = np.random.choice(all_image_paths, input_num_views, replace=False)
    
    # Generate prediction
    res = agent.generate_prediction(random_image_paths, overwrite=overwrite)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--obj_class", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--num_alternative_queries", type=int, default=2)
    parser.add_argument("--input_num_views", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    set_logger()
    
    run_vlm_segmentation(
        obj_id=args.obj_id,
        obj_class=args.obj_class,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        api_key=args.api_key,
        overwrite=args.overwrite,
        num_alternative_queries=args.num_alternative_queries,
        input_num_views=args.input_num_views
    )