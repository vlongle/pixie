import gc
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm


class CLIPArgs:
    model_name: str = "ViT-L/14@336px"
    skip_center_crop: bool = True
    batch_size: int = 64

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
            "skip_center_crop": cls.skip_center_crop,
        }




@torch.no_grad()
def extract_sam():
    pass

@torch.no_grad()
def extract_clip_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    """Extract dense patch-level CLIP features for given images"""
    from f3rm.features.clip import clip

    model, preprocess = clip.load(CLIPArgs.model_name, device=device)
    print(f"Loaded CLIP model {CLIPArgs.model_name}")

    # Patch the preprocess if we want to skip center crop
    if CLIPArgs.skip_center_crop:
        # Check there is exactly one center crop transform
        is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
        assert (
            sum(is_center_crop) == 1
        ), "There should be exactly one CenterCrop transform"
        # Create new preprocess without center crop
        preprocess = Compose(
            [t for t in preprocess.transforms if not isinstance(t, CenterCrop)]
        )
        print("Skipping center crop")

    # Preprocess the images
    images = [Image.open(path) for path in image_paths]
    raw_img_H, raw_img_W = images[0].size
    preprocessed_images = torch.stack([preprocess(image) for image in images])
    preprocessed_images = preprocessed_images.to(device)  # (b, 3, h, w)

    # Get CLIP embeddings for the images
    embeddings = []
    for i in tqdm(
        range(0, len(preprocessed_images), CLIPArgs.batch_size),
        desc="Extracting CLIP features",
    ):
        batch = preprocessed_images[i : i + CLIPArgs.batch_size]
        embeddings.append(model.get_patch_encodings(batch))
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings from flattened patches to patch height and width
    h_in, w_in = preprocessed_images.shape[-2:]
    if CLIPArgs.model_name.startswith("ViT"):
        h_out = h_in // model.visual.patch_size
        w_out = w_in // model.visual.patch_size
    elif CLIPArgs.model_name.startswith("RN"):
        h_out = max(h_in / w_in, 1.0) * model.visual.attnpool.spacial_dim
        w_out = max(w_in / h_in, 1.0) * model.visual.attnpool.spacial_dim
        h_out, w_out = int(h_out), int(w_out)
    else:
        raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
    embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    print(f"Extracted CLIP embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del model
    del preprocess
    del preprocessed_images
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings
    