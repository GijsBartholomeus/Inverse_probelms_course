"""
Data module for medical image reconstruction: sinogram
"""
import numpy as np
from skimage.transform import radon

def forward_project(image: np.ndarray, angles: np.ndarray, circle: bool = False, noise: float = 0.0) -> np.ndarray:
    """
    Compute the sinogram (Radon transform) of a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D input image array.
    angles : np.ndarray
        1D array of angles (in degrees) at which to compute projections.
    circle : bool
        Whether to assume the image is inscribed in a circle (True) or padded to square (False).
    noise : float
        Standard deviation of Gaussian noise to add to the sinogram.

    Returns
    -------
    sinogram : np.ndarray
        2D array of shape (len(image_diagonal), len(angles)) containing projection data.
    """
    # skimage.radon assumes input image is square, but handles general shapes
    sinogram = radon(image, theta=angles, circle=circle)

    # Add noise if specified
    if noise > 0:
        noise_array = np.random.normal(0, noise, sinogram.shape)
        sinogram += noise_array

    return sinogram