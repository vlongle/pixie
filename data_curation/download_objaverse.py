import multiprocessing
import objaverse
import subprocess
import glob
import pickle
import os
import hydra
from omegaconf import DictConfig
import logging
from pixie.utils import resolve_paths, load_pickle, set_logger


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Download Objaverse objects by category using Hydra configuration."""
    
    # Set up logging
    set_logger()
    
    # Resolve paths and system-specific settings
    cfg = resolve_paths(cfg)
    
    # Set the number of processes for downloading
    processes = cfg.data_curation.download.processes
    if processes is None:
        processes = multiprocessing.cpu_count()
    
    # Get download configuration
    download_cfg = cfg.data_curation.download
    
    final_dataset = load_pickle(cfg.paths.final_dataset_path)
    
    logging.info("Available categories: %s", list(final_dataset.keys()))
    
    # Determine which categories to download
    categories_to_download = list(final_dataset.keys())
    if download_cfg.obj_class:
        assert download_cfg.obj_class in final_dataset, \
            f"Category '{download_cfg.obj_class}' not found in dataset. Available categories: {list(final_dataset.keys())}"
        categories_to_download = [download_cfg.obj_class]
    
    # For each category you want to download
    for category in categories_to_download:
        logging.info("Processing category: %s", category)
        
        # Get UIDs for this category
        category_uids = final_dataset.get(category, [])
        
        if not category_uids:
            logging.warning("No objects found for category: %s", category)
            continue
        
        # Limit the number of UIDs if needed
        max_objs = download_cfg.max_objs_per_class
        if max_objs is not None:
            category_uids = category_uids[:max_objs]
            logging.info(f"Limiting {category} to {max_objs} objects (from {len(final_dataset[category])} total)")
        
        logging.info(f"Downloading {len(category_uids)} objects for category {category}")
        
        # Download the limited set of objects
        objects = objaverse.load_objects(
            uids=category_uids,
            download_processes=min(processes, len(category_uids)),
        )
        
        logging.info("Downloaded %d objects for category %s", len(category_uids), category)
        

if __name__ == "__main__":
    main()



