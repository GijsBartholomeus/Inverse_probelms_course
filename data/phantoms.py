"""
Data module for medical image reconstruction: phantoms
"""
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from PIL import Image

def generate_shepp_logan(size: int = 256) -> np.ndarray:
    """
    Generate a Shepp-Logan phantom of given size.

    Parameters
    ----------
    size : int
        Output image will be size x size pixels.

    Returns
    -------
    phantom : np.ndarray
        2D array with values in [0, 1].
    """
    base = shepp_logan_phantom()
    phantom = resize(base, (size, size), mode='reflect', anti_aliasing=True)
    return phantom


def load_image(path: str, size: tuple = None) -> np.ndarray:
    """
    Load an image from disk and convert to grayscale numpy array.

    Parameters
    ----------
    path : str
        File path to image.
    size : tuple of int, optional
        Desired output size (height, width). If None, preserves original size.

    Returns
    -------
    img : np.ndarray
        2D array with values normalized to [0, 1].
    """
    img = Image.open(path).convert('L')  # grayscale
    if size is not None:
        img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img, dtype=np.float32)
    # normalize
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr
