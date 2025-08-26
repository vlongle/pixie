import gc
from typing import List, Dict
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import cv2
from torchvision.transforms import CenterCrop, Compose, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

class CLIPSAMArgs:
    # CLIP settings
    clip_model_name: str = "ViT-L/14@336px"
    skip_center_crop: bool = True
    batch_size: int = 64
    
    # SAM settings
    sam_size: int = 1024
    obj_feat_res: int = 100  # Object-level feature resolution
    final_feat_res: int = 64  # Final output resolution
    mobilesamv2_encoder_name: str = 'mobilesamv2_efficientvit_l2'
    
    # Detector settings
    yolo_conf: float = 0.4
    yolo_iou: float = 0.9

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the model parameters."""
        return {
            "clip_model_name": cls.clip_model_name,
            "skip_center_crop": cls.skip_center_crop,
            "sam_size": cls.sam_size,
            "obj_feat_res": cls.obj_feat_res,
            "final_feat_res": cls.final_feat_res,
            "mobilesamv2_encoder_name": cls.mobilesamv2_encoder_name,
            "yolo_conf": cls.yolo_conf,
            "yolo_iou": cls.yolo_iou,
        }


def batch_iterator(batch_size: int, *args):
    """Helper for batch processing."""
    n = len(args[0])
    for i in range(0, n, batch_size):
        yield tuple(arg[i:i+batch_size] for arg in args)


def resize_image(img, longest_edge):
    """Resize image to have longest edge equal to longest_edge."""
    w, h = img.size
    if h > w:
        new_h, new_w = longest_edge, int(longest_edge * w / h)
    else:
        new_h, new_w = int(longest_edge * h / w), longest_edge
    return img.resize((new_w, new_h), Image.BILINEAR)

def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img
