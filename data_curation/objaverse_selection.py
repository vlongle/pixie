import objaverse
from sentence_transformers import SentenceTransformer, util
import torch
import pickle 
import json
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import logging
from pixie.utils import save_pickle, load_json, resolve_paths


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run objaverse object selection and categorization."""
    
    # Set up logging
    set_logger()
    
    # Resolve paths and system-specific settings
    cfg = resolve_paths(cfg)
    
    uid = objaverse.load_uids()
    logging.info("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations(uid)

    # Load category dictionary from paths config
    category_dict_path = cfg.paths.category_dict_path
    category_dict = load_json(category_dict_path)

    logging.info("Initializing sentence transformer model...")
    # Initialize model with config parameters (always use cuda when available)
    model = SentenceTransformer(cfg.data_curation.objaverse_selection.model.name, device="cuda")

    # Get object IDs and their names
    all_obj_ids = list(annotations.keys())
    all_obj_names = [annotations[obj_id].get("name", "") for obj_id in all_obj_ids]

    logging.info("Encoding object descriptions...")
    # Encode all object descriptions using config batch size
    obj_embeddings = model.encode(
        all_obj_names,
        batch_size=cfg.data_curation.objaverse_selection.batch_size,   
        convert_to_tensor=True, 
        show_progress_bar=True   
    )

    # Dictionary: object_id -> (category, similarity_score)
    assignment = {}

    top_k = cfg.data_curation.objaverse_selection.top_k

    for cat_key, cat_list in category_dict.items():
        # 1. Create the query text by combining subcategories
        cat_query = " ".join(cat_list)
        cat_embedding = model.encode(cat_query, convert_to_tensor=True)
        
        # 2. Similarities for all objects (shape: [N_objects])
        similarities = util.cos_sim(cat_embedding, obj_embeddings)[0]
        
        # 3. Get the top-k indices and their similarity values
        topk = torch.topk(similarities, k=top_k)
        topk_indices = topk.indices
        topk_values  = topk.values
        
        # 4. For each top-k object, see if it should be assigned (or reassigned)
        for i, obj_idx in enumerate(topk_indices):
            obj_id = all_obj_ids[obj_idx]
            sim_score = topk_values[i].item()
            
            # Check if this object hasn't been assigned yet OR 
            # if this category has a higher similarity than the previous assignment
            if (obj_id not in assignment) or (sim_score > assignment[obj_id][1]):
                assignment[obj_id] = (cat_key, sim_score)

    # Build final_dataset = { category: [object_ids...] }
    final_dataset = defaultdict(list)
    for obj_id, (cat_key, sim_score) in assignment.items():
        final_dataset[cat_key].append(obj_id)

    final_dataset = dict(final_dataset)

    # Save final dataset using paths config output path
    output_path = cfg.paths.final_dataset_path
    save_pickle(final_dataset, output_path)
    
    logging.info(f"Final dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
