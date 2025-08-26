import os
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, Optional
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
from pixie.utils import resolve_paths, load_json, save_json, get_vlm_api_key, set_logger
from vlmx.agent import Agent, AgentConfig
from PIL import Image


class DataFilteringAgent(Agent):
    """Agent for filtering rendered images based on appropriateness."""
    
    OUT_RESULT_PATH = "vlm_results.json"
    
    def __init__(self, cfg: AgentConfig, class_name: str, search_terms: str):
        super().__init__(cfg)
        self.class_name = class_name
        self.search_terms = search_terms
    
    def _make_system_instruction(self):
        return f"""
        We need to select some images of the classes: {self.class_name}. This class includes objects like {self.search_terms}. 
        We will provide you some image rendered from the 3d model. You need to either return True or False. 
        
        Return False to reject the image as inappropriate for the video game development. Some common reasons for rejection:
        - The image doesn't clearly depict the object class
        - The image is too dark or too bright or too blurry or has some low other qualities. Remember, we want high-quality training data.
        - The image contains other things in addition to the object. REMEMBER, we only want images that depict cleanly ONE SINGLE OBJECT belong to one of the classes. 
          But you also need to use your common sense and best judgement. For example, for class like "flowers", the object might include a vase of flowers 
          (you rarely see a single flower in the wild). So you should return True in this case.
        - We do want diversity in our dataset collection. So even if the texture of the object is a bit un-usual, as long as you can recognize it as 
          belonging to the class / search terms, you should return True. Only remove low-quality assets.

        We'll be using the 3d models to learn physic parameters like material and young modulus to simulate the physics of the object.
        E.g., the tree swaying in the wind or thing being dropped from a height. Therefore, you need to decide if the image depicts an object 
        that is likely to be used in a physics simulation.

        The return format is
        ```json
        {{
        "is_appropriate": true (or false),
        "reason": "reason for the decision"
        }}
        ```
        """

    def _make_prompt_parts(self, image_path: str):
        return ["The image is :", Image.open(image_path)]

    def parse_response(self, response):
        json_str = response.text.strip().strip("```json").strip()
        parsed_response = json.loads(json_str, strict=False)
        save_json(parsed_response, os.path.join(self.cfg.out_dir, self.OUT_RESULT_PATH))
        return parsed_response


def get_rendered_images(render_base_dir: str, obj_class: Optional[str] = None) -> Dict[str, str]:
    """Get dictionary mapping object IDs to their rendered image paths."""
    result_dict = {}
    
    # Get categories to process
    if obj_class:
        categories = [obj_class]
        base_path = os.path.join(render_base_dir, obj_class)
    else:
        categories = [os.path.basename(f.rstrip('/')) for f in sorted(glob.glob(f"{render_base_dir}/*/"))]
    
    # Iterate through categories
    for category in categories:
        category_path = os.path.join(render_base_dir, category) if not obj_class else base_path
        
        # Get all object folders (UIDs)
        for obj_folder in sorted(glob.glob(f"{category_path}/*/")):
            obj_id = os.path.basename(obj_folder.rstrip('/'))
            
            # Find PNG files
            png_files = sorted(glob.glob(f"{obj_folder}/*.png"))
            if png_files:
                unique_id = f"{category}/{obj_id}"
                result_dict[unique_id] = png_files[0]  # Use first image
    
    return result_dict


def process_single_image(args: Tuple) -> Tuple[str, dict]:
    """Process a single image with VLM."""
    obj_id, image_path, output_dir, api_key, class_name, search_terms, overwrite, model_name = args
    
    try:
        agent = DataFilteringAgent(
            AgentConfig(
                model_name=model_name,
                out_dir=os.path.join(output_dir, obj_id),
                api_key=api_key
            ),
            class_name=class_name,
            search_terms=search_terms
        )
        
        agent.generate_prediction(image_path, overwrite=overwrite)
        response = agent.load_prediction()
        return obj_id, response
        
    except Exception as e:
        logging.error(f"Error processing {obj_id}: {e}")
        return obj_id, {
            "is_appropriate": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        }


def analyze_results(results: Dict[str, dict]) -> None:
    """Analyze and print VLM filtering results."""
    total = len(results)
    successful = sum(1 for r in results.values() if r.get("is_appropriate") is not None)
    errors = sum(1 for r in results.values() if r.get("is_appropriate") is None)
    appropriate = sum(1 for r in results.values() if r.get("is_appropriate") is True)
    inappropriate = sum(1 for r in results.values() if r.get("is_appropriate") is False)
    
    # Print overall statistics
    logging.info("\nProcessing Statistics:")
    logging.info(f"Total images: {total}")
    logging.info(f"Successfully processed: {successful} ({successful/total*100:.1f}%)")
    logging.info(f"Processing errors: {errors} ({errors/total*100:.1f}%)")
    
    if successful > 0:
        logging.info(f"Appropriate: {appropriate} ({appropriate/successful*100:.1f}% of successful)")
        logging.info(f"Inappropriate: {inappropriate} ({inappropriate/successful*100:.1f}% of successful)")
    
    # Analyze by category
    stats_by_category = {}
    for obj_id, result in results.items():
        category = obj_id.split('/')[0]
        if category not in stats_by_category:
            stats_by_category[category] = {"total": 0, "appropriate": 0, "inappropriate": 0, "errors": 0}
        
        stats_by_category[category]["total"] += 1
        if result.get("is_appropriate") is True:
            stats_by_category[category]["appropriate"] += 1
        elif result.get("is_appropriate") is False:
            stats_by_category[category]["inappropriate"] += 1
        else:
            stats_by_category[category]["errors"] += 1
    
    logging.info("\nResults by category:")
    for category, stats in stats_by_category.items():
        logging.info(f"  {category}:")
        logging.info(f"    Total: {stats['total']}")
        if stats['total'] > 0:
            logging.info(f"    Appropriate: {stats['appropriate']} ({stats['appropriate']/stats['total']*100:.1f}%)")
            logging.info(f"    Inappropriate: {stats['inappropriate']} ({stats['inappropriate']/stats['total']*100:.1f}%)")
            logging.info(f"    Errors: {stats['errors']} ({stats['errors']/stats['total']*100:.1f}%)")


def process_category(category: str, rendered_images: Dict[str, str], cfg: DictConfig, 
                    filter_cfg: DictConfig, category_dict: dict, api_key: str) -> None:
    """Process a single category of images."""
    # Set output directory for this category
    output_dir = os.path.join(cfg.paths.vlm_filtering_results_dir, category)
    
    # Check if already processed and not overwriting
    results_file = os.path.join(output_dir, "all_results.json")
    if os.path.exists(results_file) and not filter_cfg.overwrite:
        logging.info(f"Category {category} already processed, skipping...")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get search terms for this category
    search_terms = " ".join(category_dict.get(category, []))
    
    # Build process arguments for this category
    process_args = [
        (obj_id, image_path, output_dir, api_key, category,
         search_terms, filter_cfg.overwrite, filter_cfg.model_name)
        for obj_id, image_path in rendered_images.items()
    ]
    
    logging.info(f"Processing {len(process_args)} images for category: {category}")
    
    # Process in parallel
    num_workers = filter_cfg.num_workers or os.cpu_count()
    results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for obj_id, response in tqdm(
            executor.map(process_single_image, process_args),
            total=len(process_args),
            desc=f"Processing {category}"
        ):
            results[obj_id] = response
    
    # Save results
    save_json(results, results_file)
    
    # Save error details if any
    error_details = {
        obj_id: result for obj_id, result in results.items() 
        if result.get("is_appropriate") is None
    }
    if error_details:
        save_json(error_details, os.path.join(output_dir, "error_details.json"))
    
    # Log category statistics
    successful = sum(1 for r in results.values() if r.get("is_appropriate") is not None)
    appropriate = sum(1 for r in results.values() if r.get("is_appropriate") is True)
    logging.info(f"Category {category}: {appropriate}/{successful} appropriate")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to filter rendered images using VLM."""
    set_logger()
    cfg = resolve_paths(cfg)
    
    filter_cfg = cfg.data_curation.vlm_filtering
    
    # Load category dictionary
    category_dict = load_json(cfg.paths.category_dict_path)
    
    categories_to_process = list(category_dict.keys())
    # Determine categories to process
    if filter_cfg.obj_class:
        assert filter_cfg.obj_class in category_dict, \
            f"Category '{filter_cfg.obj_class}' not found in category dictionary"
        categories_to_process = [filter_cfg.obj_class]
    
    # Handle analyze_only mode
    if filter_cfg.analyze_only:
        all_results = {}
        for category in categories_to_process:
            results_file = os.path.join(cfg.paths.vlm_filtering_results_dir, 
                                       category, "all_results.json")
            if os.path.exists(results_file):
                category_results = load_json(results_file)
                all_results.update(category_results)
            else:
                logging.warning(f"No results found for category: {category}")
        
        analyze_results(all_results)
        return
    
    # Get API key
    api_key = get_vlm_api_key(cfg, filter_cfg.model_name)
    assert api_key, "No API key found for VLM"
    
    # Process each category
    for category in categories_to_process:
        logging.info(f"\nProcessing category: {category}")
        
        # Get rendered images for this category
        rendered_images = get_rendered_images(
            cfg.paths.render_outputs_base_dir,
            category  # Process one category at a time
        )
        
        if not rendered_images:
            logging.warning(f"No rendered images found for category: '{category}' in '{cfg.paths.render_outputs_base_dir}'")
            continue
        
        process_category(category, rendered_images, cfg, filter_cfg, 
                        category_dict, api_key)
    
    # After processing all categories, show combined statistics
    logging.info("\n" + "="*50)
    logging.info("="*50)
    
    all_results = {}
    for category in categories_to_process:
        results_file = os.path.join(cfg.paths.vlm_filtering_results_dir, 
                                   category, "all_results.json")
        if os.path.exists(results_file):
            category_results = load_json(results_file)
            all_results.update(category_results)
    
    analyze_results(all_results)


if __name__ == "__main__":
    main()