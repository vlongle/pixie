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

CRITIC_SYSTEM_INSTRUCTION = """
You are a segmentation quality critic. Your task is to evaluate the quality of segmentation results produced by a CLIP-based segmentation model.

You will be shown:
1. A set of original RGB images of a 3D object from different views
2. Segmentation results for different part queries

Your job is to:
1. Evaluate each segmentation query based on how well it separates the object into meaningful parts
2. Score each query on a scale of 1-10 (10 being perfect)
3. Provide reasoning for your scores
4. Suggest improvements to the queries if needed

Consider the following factors in your evaluation:
- Does the segmentation properly separate the object into distinct, semantically meaningful parts?
- Are the boundaries of the segments accurate and clean?
- Is any important part of the object missed or incorrectly segmented?
- IMPORTANT: note that our imperfect CLIP segmentation model is heavily dependent on the choice of part queries. Thus,
even if a query might not be semantically correct, as long as it is useful for separating the object into distinct parts,
you should score it high.
- Bad queries would result in bad segmentation that are noisy or different parts are not correctly and/or clearly separated.

Your output should be a JSON in the following format:

```json
{
  "query_evaluations": {
    "query_0": {
      "score": 8,
      "reasoning": "This query effectively separates the object into functionally distinct parts. The boundaries are clean and consistent across different views."
    },
    "query_1": {
      "score": 3,
      "reasoning": "This query fails to distinguish important parts of the object, making it unsuitable for physical property assignment."
    },
    ...
  },
  "best_query": "query_1",
  "suggested_improvements": "Consider using more specific terms like 'ceramic pot' instead of just 'pot' to improve segmentation boundaries."
}
```
where `query_{i}` is the i-th query in the "all_queries" list.

Be detailed in your reasoning and make concrete suggestions for improvements.
"""

from typing import List, Dict

class SegmentationCriticAgent(Agent):
    OUT_RESULT_PATH = "vlm_critic_results.json"
    
    def _make_system_instruction(self):
        return CRITIC_SYSTEM_INSTRUCTION 

    def _make_prompt_parts(self, rgb_image_paths: List[str], query_segmentation_paths: Dict[str, List[str]]):
        prompt_parts = ["I'll show you original RGB images of an object and then segmentation results for different queries."]
        
        # Add original RGB images
        prompt_parts.append("Original RGB images of the object:")
        rgb_images = [Image.open(img_path) for img_path in rgb_image_paths]
        prompt_parts.extend(rgb_images)
        
        # Add segmentation results for each query
        prompt_parts.append("Now I'll show you segmentation results for different queries:")
        
        for query, seg_paths in query_segmentation_paths.items():
            prompt_parts.append(f"Segmentation results for query: {query}")
            seg_images = [Image.open(img_path) for img_path in seg_paths]
            prompt_parts.extend(seg_images)
        
        return prompt_parts

    def parse_response(self, response):
        try:
            # Extract JSON from the response
            response_text = response.text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            # Save the results
            save_json(parsed_response, os.path.join(self.cfg.out_dir, self.OUT_RESULT_PATH))
            return parsed_response
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            logging.error(f"Response: {response.text}")
            return {"error": str(e), "raw_response": response.text}


def run_vlm_seg_critic(obj_id: str, data_dir: str, vlm_seg_results_dir: str, 
                      vlm_seg_critic_results_dir: str, input_num_views: int = 15,
                      model_name: str = 'gemini-2.5-pro-preview-03-25', 
                      api_key: str = '', overwrite: bool = False):
    """Run VLM segmentation critic to evaluate segmentation quality."""
    
    # Setup paths
    rgb_image_path = os.path.join(data_dir, obj_id, "train")
    segmentation_base_path = os.path.join(vlm_seg_results_dir, obj_id)
    output_dir = os.path.join(vlm_seg_critic_results_dir, obj_id)
    
    # Ensure necessary paths exist
    if not os.path.exists(rgb_image_path):
        logging.error(f"RGB image path does not exist: {rgb_image_path}")
        return
    
    if not os.path.exists(segmentation_base_path):
        logging.error(f"Segmentation base path does not exist: {segmentation_base_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB image paths
    all_rgb_paths = glob.glob(os.path.join(rgb_image_path, "*.png"))
    if len(all_rgb_paths) < input_num_views:
        logging.warning(f"Not enough RGB images found. Found {len(all_rgb_paths)}, needed {input_num_views}")
        if len(all_rgb_paths) == 0:
            return
        input_num_views = len(all_rgb_paths)
    
    selected_rgb_paths = np.random.choice(all_rgb_paths, input_num_views, replace=False)
    
    # Get query segmentation paths
    query_pattern = "query_*"
    query_folders = glob.glob(os.path.join(segmentation_base_path, query_pattern))
    query_folders = [os.path.basename(folder) for folder in query_folders if os.path.isdir(folder)]
    
    logging.info(f"Found query folders: {query_folders}")
    
    if not query_folders:
        logging.error(f"No query folders found in {segmentation_base_path}")
        return
    
    # Build dictionary of query to segmentation image paths
    query_to_segmentation_paths = {}
    for query in query_folders:
        seg_path = os.path.join(segmentation_base_path, query)
        seg_images = glob.glob(os.path.join(seg_path, "*.png"))
        if seg_images:
            query_to_segmentation_paths[query] = seg_images
    
    if not query_to_segmentation_paths:
        logging.error("No segmentation images found for any query")
        return
    
    # Initialize agent and evaluate
    agent = SegmentationCriticAgent(AgentConfig(
        model_name=model_name,
        out_dir=output_dir,
        api_key=api_key,
    ))
    
    agent.generate_prediction(
        selected_rgb_paths, 
        query_to_segmentation_paths,
        overwrite=overwrite
    )
    results = agent.load_prediction()
    
    # Print results summary
    if "best_query" in results:
        logging.info(f"Best query: {results['best_query']}")
        if results['best_query'] in results.get('query_evaluations', {}):
            score = results['query_evaluations'][results['best_query']]['score']
            logging.info(f"Score: {score}/10")
    
    logging.info(f"Full evaluation results saved to {os.path.join(output_dir, agent.OUT_RESULT_PATH)}")
    return results


if __name__ == "__main__":
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    parser.add_argument('--obj_id', type=str, required=True, help='Object ID to evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--vlm_seg_results_dir', type=str, required=True, help='VLM segmentation results directory')
    parser.add_argument('--vlm_seg_critic_results_dir', type=str, required=True, help='VLM segmentation critic results directory')
    parser.add_argument('--input_num_views', type=int, default=15, help='Number of RGB samples to use')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing results')
    parser.add_argument('--model_name', type=str, default='gemini-2.5-pro-preview-03-25')
    parser.add_argument('--api_key', type=str, default='')
    args = parser.parse_args()
    set_logger()
    
    run_vlm_seg_critic(
        obj_id=args.obj_id,
        data_dir=args.data_dir,
        vlm_seg_results_dir=args.vlm_seg_results_dir,
        vlm_seg_critic_results_dir=args.vlm_seg_critic_results_dir,
        input_num_views=args.input_num_views,
        model_name=args.model_name,
        api_key=args.api_key,
        overwrite=args.overwrite
    )