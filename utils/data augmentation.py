"""A file that contains functions that apply data augmentations.

1. `extract_random_patches`: Extracts random patches from an image.
"""
import numpy as np


def extract_random_patches(img, patch_size=20, num_patches=5):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(h, w) // 3  # radius for sampling

    patches = []
    for _ in range(num_patches):
        while True:
            dx = np.random.randint(-r, r)
            dy = np.random.randint(-r, r)
            if dx**2 + dy**2 <= r**2:
                x = cx + dx - patch_size // 2
                y = cy + dy - patch_size // 2
                if 0 <= x < w - patch_size and 0 <= y < h - patch_size:
                    patch = img[y:y+patch_size, x:x+patch_size]
                    patches.append(patch)
                    break
    return patches



def random_flip_rotate_np(img, rng=None):
    """
    img  : H×W or H×W×C  NumPy array
    rng  : np.random.Generator (optional)
    • 50 % chance horizontal flip
    • 50 % chance vertical flip
    • random 0/90/180/270° rotation
    """
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:          # H-flip
        img = np.flip(img, axis=1)
    if rng.random() < 0.5:          # V-flip
        img = np.flip(img, axis=0)
    k = rng.integers(0, 4)          # rotate k×90°
    img = np.rot90(img, k, axes=(0, 1))
    return img.copy()               # ensure contiguous memory

