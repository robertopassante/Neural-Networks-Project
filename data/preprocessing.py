import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Colors defined by the DeepGlobe dataset for masks
COLOR_TO_CLASS = {
    (0, 255, 255): 0,  # Urban land
    (255, 255, 0): 1,  # Agriculture land
    (255, 0, 255): 2,  # Rangeland
    (0, 255, 0): 3,    # Forest land
    (0, 0, 255): 4,    # Water
    (255, 255, 255): 5,# Barren land
    (0, 0, 0): 6       # Unknown / Ignore
}

def rgb_to_class(mask_img):
    """
    Converts an RGB mask (PIL Image) to a single-channel long tensor 
    representing class indices.
    """
    mask_np = np.array(mask_img)
    # Initialize with class 6 (Unknown)
    mask_class = np.full(mask_np.shape[:2], 6, dtype=np.int64)
    for rgb, class_idx in COLOR_TO_CLASS.items():
        # Find pixels exactly matching this color
        matches = (mask_np == np.array(rgb)).all(axis=-1)
        mask_class[matches] = class_idx
    return mask_class

def get_transforms():
    """
    Returns the standard ImageNet-based normalization transforms for images.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
