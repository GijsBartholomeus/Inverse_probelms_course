"""
Metrics for evaluating image reconstruction quality.
"""
import numpy as np


def mse(img1, img2):
    """
    Calculate the Mean Squared Error between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.

    Returns
    -------
    float
        The Mean Squared Error.
    """
    return np.mean((img1 - img2) ** 2)


def psnr(img1, img2, max_val=None):
    """
    Calculate the Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.
    max_val : float, optional
        The maximum possible pixel value. If None, the maximum value 
        of img1 is used.

    Returns
    -------
    float
        The Peak Signal-to-Noise Ratio in dB.
    """
    if max_val is None:
        max_val = img1.max()
    
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    
    return 20 * np.log10(max_val / np.sqrt(mse_val))


def ssim(img1, img2):
    """
    Calculate the Structural Similarity Index between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.

    Returns
    -------
    float
        The Structural Similarity Index.
    """
    # Placeholder for future implementation
    # SSIM requires more complex calculations including local means, variances, and covariance
    return 0.0
