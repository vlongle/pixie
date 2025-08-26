import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging
import objaverse
from pixie.utils import resolve_paths, load_pickle, run_cmd, set_logger


def render_object(args):
    """Render a single object."""
    uid, output_dir, gpu_id, blender_cmd, resolution, views = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Skip if already rendered
    if glob.glob(f"{output_dir}/*.png"):
        return True
        
    os.makedirs(output_dir, exist_ok=True)
    # Use --obj_id instead of --obj with path
    cmd = f'{blender_cmd} -b -P data_curation/render_blender.py -- --obj_id {uid} --output {output_dir} --views {views} --resolution {resolution}'
    
    return run_cmd(cmd, step_name=f"Render {uid}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_logger()
    cfg = resolve_paths(cfg)
    
    # Load config
    render_cfg = cfg.data_curation.rendering
    dataset = load_pickle(cfg.paths.final_dataset_path)
    
    # Get categories to render
    categories = [render_cfg.obj_class] if render_cfg.obj_class else list(dataset.keys())
    assert not render_cfg.obj_class or render_cfg.obj_class in dataset, \
        f"Category {render_cfg.obj_class} not found"
    
    # Prepare render jobs
    render_jobs = []
    for category in categories:
        uids = dataset[category][:render_cfg.max_objs_per_class]
        
        for i, uid in enumerate(uids):
            output_dir = os.path.join(cfg.paths.render_outputs_base_dir, 
                                     category,
                                     uid).rstrip('/')
            
            gpu_id = i % render_cfg.num_gpus
            render_jobs.append((uid, output_dir, gpu_id, 
                               cfg.paths.blender_path, render_cfg.resolution, 
                               render_cfg.views))
    
    # Render in parallel
    logging.info(f"Rendering {len(render_jobs)} objects")
    with ProcessPoolExecutor(max_workers=render_cfg.num_gpus * render_cfg.jobs_per_gpu) as executor:
        futures = [executor.submit(render_object, job) for job in render_jobs]
        results = []
        for future in tqdm(futures, total=len(futures)):
            try:
                result = future.result(timeout=render_cfg.timeout)
                results.append(result)
            except TimeoutError:
                logging.warning(f"Render job timed out after {render_cfg.timeout} seconds")
                results.append(False)
    
    logging.info(f"Rendered {sum(results)}/{len(render_jobs)} successfully")


if __name__ == "__main__":
    main()