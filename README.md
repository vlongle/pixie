<div align="center">
  <br>
  <br>
  <h1>Pixie: Physics from Pixels</h1>
</div>

<p align="center">
  <a href="https://pixie-3d.github.io/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2508.17437">
    <img alt="Arxiv paper link" src="https://img.shields.io/badge/arxiv-2508.17437-blue">
  </a>
    <!-- <a href="https://huggingface.co/datasets/vlongle/pixie/tree/main">
    <img alt="Model Checkpoints link" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a> -->
  <a href="https://x.com/LongLeRobot/status/1961139689886552481">
    <img alt="Twitter Thread" src="https://img.shields.io/badge/Twitter-Thread-1DA1F2">
  </a>
</p>

<div align="center">

**[Long Le](https://vlongle.github.io/)**$^1$ · **[Ryan Lucas](https://ryanlucas3.github.io/)**$^2$ · **[Chen Wang](https://cwchenwang.github.io/)**$^1$ · **[Chuhao Chen](https://czzzzh.github.io/)**$^1$ · **[Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/)**$^1$ · **[Eric Eaton](https://www.seas.upenn.edu/~eeaton/)**$^1$ · **[Lingjie Liu](https://lingjie0206.github.io/)**$^1$

$^1$ University of Pennsylvania · $^2$ MIT

</div>


<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="docs/assets/teaser_full_high_quality.gif">

Photorealistic 3D reconstructions (NeRF, Gaussian Splatting) capture geometry & appearance but **lack physics**. This limits 3D reconstruction to static scenes. Recently, there has been a surge of interest in integrating physics into 3D modeling. But existing test‑time optimisation methods are slow and scene‑specific. **Pixie** trains a neural network that maps pretrained visual features (i.e., CLIP) to **dense material fields** of physical properties in a single forward pass, enabling fast and generalizable physics inference and simulation.

## 💡 Contents

1. [Installation](#installation)
2. [Download Pre-trained Models](#download-models)
3. [Usage](#usage)
4. [VLM Labeling](#vlm-labeling)
5. [Training](#training)
6. [Common Issues](#common-issues)
7. [Citation](#citation)



<h2 id="installation">⚙️ Installation</h2>

```
git clone git@github.com:vlongle/pixie.git
conda create -n pixie python=3.10
conda activate pixie
pip install -e .
```
Install `torch` and `torchvision` according to your cuda version (e.g., 11.8, 12.1) and the [official instruction](https://pytorch.org/).
Install additional dependencies for f3rm (NeRF CLIP distilled feature field):

```
# ninja so compilation is faster!
pip install ninja 
# Install tinycudann (may take a while)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install third-party packages
pip install -e third_party/nerfstudio
pip install -e third_party/f3rm

# Install PyTorch3D and other dependencies
pip install -v "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install viser==0.2.7
pip install tyro==0.6.6
```

Install PhysGaussian dependencies (for MPM simulation)

```
pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/simple-knn/
pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/diff-gaussian-rasterization/
```
Install VLM utils 

```
pip install -e third_party/vlmx
```
Install FlashAttention to use [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

```
MAX_JOBS=16 pip install -v -U flash-attn --no-build-isolation
```

Install dependencies / add-ons for Blender. We use [Blender 4.3.2](https://www.blender.org/download/).
1. Install [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF) add-on and set `paths.blender_nerf_addon_path` to BlenderNeRF's zip file.
2. Install python packages for Blender. Replace the path by your actual Blender path
    ```
    /home/{YOUR_USERNAME}/blender/blender-4.3.2-linux-x64/4.3/python/bin/python3.11 -m pip install objaverse
    ```
Install the [Gaussian-Splatting addon](https://github.com/ReshotAI/gaussian-splatting-blender-addon) and set [paths.blender_gs_addon_path](config/paths/default.yaml) in the config.


Set the appropriate api keys and select VLM models you'd like in [config/segmentation/default.yaml](config/segmentation/default.yaml), we support OpenAI, Claude, Google's Gemini, or Qwen (local, no api needed). You can also implement more model wrappers yourself following our template!

<h2 id="download-models">📥 Download Pre-trained Models and Data</h2>

We provide pre-trained model checkpoints via HuggingFace Datasets. To download the models:

```bash
python scripts/download_data.py
```

<h2 id="usage">🎯 Usage</h2>

### Synthetic Objaverse

```
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 [physics.save_ply=false] [material_mode={vlm,neural}]
```

`save_ply=true` is slower, only used for rendering fancy phyiscs simulation in Blender. `material_mode=vlm` uses VLM for labeling the data based on our in-context tuned examples. This is how we generate our dataset! `material_mode=neural` uses our trained neural networks to produce physics predictions.

This code will:
1. Download the objaverse asset `obj_id`
2. Render it in Blender using `rendering.num_images` (default 200)
3. Train a NeRF distilled CLIP field using `training_3d.nerf.max_iterations`
4. Train a gaussian splatting model using `training_3d.gaussian_splatting.max_iterations`
5. Generate a voxel feature grid from the CLIP field
6. Either
    - Apply the material dictionary predicted by a VLM (for generating data to train our model) `material_mode=vlm`
    - Use our trained UNet model to predict the physics field `material_mode=neural`. 
7. Run the MPM physics solver using the physics parameters.


Run
```
python render.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8
```
for fancy rendering in Blender.

Check the outputs in the notebook: [nbs/pixie.ipynb](nbs/pixie.ipynb).

### Real Scene

For real scene, run 
```
python pipeline.py \
    is_objaverse_object=false \
    obj_id=bonsai \
    material_mode=neural \
    paths.data_dir='${paths.base_path}/real_scene_data' \
    paths.outputs_dir='${paths.base_path}/real_scene_models' \
    paths.render_outputs_dir='${paths.base_path}/real_scene_render_outputs' \
    training.enforce_mask_consistency=false
```
Use `segmentation.neural.cache_results=true` if the latest inferene already contains `obj_id`.

Check the outputs in the notebook: [nbs/real_scene.ipynb](nbs/real_scene.ipynb).

<h2 id="vlm-labeling">🏷️ VLM Labeling</h2>

1. Compute the cosine similarity between each Objaverse object name to an object class we'd like (e.g., `tree`) and keep the `top_k` for our PixieVerse dataset.
    ```
    python data_curation/objaverse_selection.py
   ```
2. Download objaverse assets
    ```
    python data_curation/download_objaverse.py [data_curation.download.obj_class=tree] 
   ```
3. Render 1 view per object
    ```
    python data_curation/render_objaverse_classes.py [data_curation.rendering.obj_class=tree] [data_curation.rendering.max_objs_per_class=1] [data_curation.rendering.timeout=80]
    ```
    Then use VLM to filter out low-quality assets
    ```
      python pixie/vlm_labeler/vlm_data_filtering.py [data_curation.vlm_filtering.obj_class=tree]
    ```
4. Manual filtering
VLM does a decent job but not perfect. We run
    ```
    streamlit run data_curation/manual_data_filtering_correction.py [data_curation.manual_correction.obj_class=tree]
    ```
    which creates a web browser with the discarded images and the chosen images by VLM. You can skim through them quickly and tick the checkbox to flip the label and correct the VLM. Then, click "save_changes", this creates `all_results_corrected.json` which is basically `all_results.json` but which the checked boxes objects flipped.

<h2 id="training">🎓 Training</h2>

1. Compute the normalization.
    ```
    python third_party/Wavelet-Generation/data_utils/inspect_ranges.py
    ```
2. Train the discrete and continuous 3D UNet models

    Train discrete:
    ```
    python third_party/Wavelet-Generation/trainer/training_discrete.py
    ```
    Train continuous:
    ```
    python third_party/Wavelet-Generation/trainer/training_continuous_mse.py
    ```
    Adjust [training.training.batch_size](config/training/default.yaml) and other params as needed. We used 6 NVIDIA RTX A6000 GPU (~49 GB) for training each model with 128 CPUs and 450 GBs of RAM. Adjust your `batch_size` and `data_worker` according to your resource availability.
3. Then run inference
    ```
    python third_party/Wavelet-Generation/trainer/inference_combined.py [obj_id=8e24a6d4d15c4c62ae053cfa67d99e67]
    ```
    If `obj_id` not provided, we will evaluate on the entire test set.
4. Map the predicted voxel grid to world coordinate and interpolate to gaussian splatting, then run physics simulation. Taken care of by `pipeline.py`:
    ```
    python pipeline.py material_mode=neural obj_id=... [segmentation.neural.result_id='"YOUR_RESULT_TIME_STAMP"'] [segmentation.neural.feature_type=clip]
    ```




<h2 id="common-issues">💀 Common Issues</h2>

If you ran into `UnicodeEncodeError: 'ascii' codec can't encode characters in position`, try to re-install warp_lang:

```
pip install --force-reinstall warp_lang==0.10.1
```

If you ran into `ValueError: numpy.dtype size changed, may indicate binary incompatibility`, try to re-install numpy:

```
pip install --force-reinstall numpy==1.24.4
```
If you run into issues installing `tinycudann`, try installing from source via `git clone ` following [their instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension).

If you run into issue installing  gaussian-splatting submodules:
```
pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/simple-knn/
pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/diff-gaussian-rasterization/
```
Try installing without the `-e` flag.

## 😊 Acknowledgement
We would like to thank the authors of [PhysGaussian](https://xpandora.github.io/PhysGaussian/), [F3RM](https://github.com/f3rm/f3rm), [Wavelet Generation](https://github.com/edward1997104/Wavelet-Generation), [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)  and others for releasing their source code.


<h2 id="citation">📚 Citation</h2>

If you find this codebase useful, please consider citing:

```bibtex
@article{le2025pixie,
  title={Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels},
  author={Le, Long and Lucas, Ryan and Wang, Chen and Chen, Chuhao and Jayaraman, Dinesh and Eaton, Eric and Liu, Lingjie},
  journal={arXiv preprint arXiv:2508.17437},
  year={2025}
}
```

