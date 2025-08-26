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

from pixie.utils import str2bool, set_logger
import numpy as np


def sample_value(range_list):
    """Sample a random value from within a given range.
    
    Args:
        range_list (list): List containing [min_value, max_value]
    
    Returns:
        float: Randomly sampled value within the range
    """
    min_val, max_val = range_list
    return min_val + (max_val - min_val) * np.random.random()


def evaluate_constraint(constraint: str, material_dict: dict) -> bool:
    """Evaluate a single constraint using the material dictionary.
    
    Args:
        constraint (str): Python code string containing the constraint
        material_dict (dict): Dictionary containing material properties
        
    Returns:
        bool: True if constraint is satisfied, False otherwise
    """
    # Strip markdown code block markers
    constraint = constraint.strip().replace('```python', '').replace('```', '').strip()
    try:
        exec(constraint)
        return True
    except AssertionError:
        return False
    except Exception as e:
        logging.warning(f"Error evaluating constraint: {e}")
        return False


def sample_material_dict(material_ranges: dict) -> dict:
    """Sample values for all material properties within their ranges.
    
    Args:
        material_ranges (dict): Dictionary containing ranges for material properties
        
    Returns:
        dict: Dictionary with sampled values
    """
    sampled_dict = {}
    for part_name, properties in material_ranges.items():
        sampled_dict[part_name] = {
            "density": sample_value(properties["density"]),
            "E": sample_value(properties["E"]),
            "nu": sample_value(properties["nu"]),
            "material_id": properties["material_id"]  # material_id is discrete, no sampling needed
        }
    return sampled_dict


def check_all_constraints(material_dict: dict, constraints: list) -> bool:
    """Check if all constraints are satisfied for the given material dictionary.
    
    Args:
        material_dict (dict): Dictionary containing sampled material properties
        constraints (list): List of constraint strings
        
    Returns:
        bool: True if all constraints are satisfied, False otherwise
    """
    # return all(evaluate_constraint(constraint, material_dict) for constraint in constraints)
    return evaluate_constraint(constraints, material_dict)


class MathSamplerAgent(Agent):
    OUT_RESULT_PATH = "vlm_results.json"
    def _make_system_instruction(self):
        return "..."

    def generate_prediction(self, json_path: str, overwrite: bool = False, max_attempts: int = 1000):
        """Generate valid material properties through rejection sampling.
        
        Args:
            json_path (str): Path to JSON file containing ranges and constraints
            overwrite (bool): Whether to overwrite existing results
            max_attempts (int): Maximum number of sampling attempts before giving up
            
        Returns:
            dict: Dictionary containing valid sampled material properties
        """
        # Load input data
        out_path = os.path.join(self.cfg.out_dir, self.OUT_RESULT_PATH)
        if (
            os.path.exists(out_path)
            and not overwrite
        ):
            logging.info(
                f"{self.__class__.__name__}: Prediction already exists at {out_path}. Skipping generation."
            )
            return self.load_prediction()
        input_data = load_json(json_path)
        logging.info(f">> Input data: {input_data}")
        material_ranges = input_data["material_dict"]
        constraints = input_data["constraints"]
        # print("Constraints: ", constraints)
        
        # Rejection sampling
        for attempt in range(max_attempts):
            sampled_dict = sample_material_dict(material_ranges)
            if check_all_constraints(sampled_dict, constraints):
                result = {
                    "material_dict": sampled_dict,
                    "sampling_attempts": attempt + 1
                }
                logging.info(f"succeed after {attempt + 1} attempts")
                return self.parse_response(type('Response', (), {'text': json.dumps(result)}))
        
        raise RuntimeError(f"Failed to find valid sample after {max_attempts} attempts")

    def parse_response(self, response):
        json_str = response.text.strip().strip("```json").strip()
        parsed_response = json.loads(json_str, strict=False)
        save_json(parsed_response, os.path.join(
        self.cfg.out_dir, self.OUT_RESULT_PATH))
        return parsed_response


def run_vlm_phys_sampler(obj_id: str, vlm_seg_results_dir: str, 
                        vlm_seg_mat_sample_results_dir: str, 
                        num_sample_mat: int = 5, model_name: str = "gemini-1.5-flash-latest",
                        api_key: str = "", overwrite: bool = False):
    """Run VLM physics sampler to generate material samples."""
    
    seed_everything(0)

    # Get VLM results path
    json_path = os.path.join(vlm_seg_results_dir, obj_id, "vlm_results.json")
    if not os.path.exists(json_path):
        logging.error(f"VLM results not found at {json_path}")
        return

    # Generate material samples
    for i in range(num_sample_mat):
        logging.info(f">> Sampling material for {obj_id} sample {i}")
        output_dir = os.path.join(vlm_seg_mat_sample_results_dir, obj_id, f"sample_{i}")
        os.makedirs(output_dir, exist_ok=True)
        
        agent = MathSamplerAgent(AgentConfig(
            model_name=model_name,
            out_dir=output_dir,
            api_key=api_key,
        ))
        
        res = agent.generate_prediction(json_path, overwrite=overwrite)
        logging.info(f"Generated material sample {i} for {obj_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--vlm_seg_results_dir", type=str, required=True)
    parser.add_argument("--vlm_seg_mat_sample_results_dir", type=str, required=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--num_sample_mat", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash-latest")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    
    set_logger()
    run_vlm_phys_sampler(
        obj_id=args.obj_id,
        vlm_seg_results_dir=args.vlm_seg_results_dir,
        vlm_seg_mat_sample_results_dir=args.vlm_seg_mat_sample_results_dir,
        num_sample_mat=args.num_sample_mat,
        model_name=args.model_name,
        api_key=args.api_key,
        overwrite=args.overwrite
    )
