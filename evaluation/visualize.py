"""
Visualization tools for medical image reconstruction.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_image(img, title=None, cmap='gray'):
    """
    Plot an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to plot.
    title : str, optional
        The title of the plot.
    cmap : str, optional
        The colormap to use, by default 'gray'.

    Returns
    -------
    matplotlib.pyplot.Figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig


def plot_error_map(img1, img2, title=None, cmap='viridis'):
    """
    Plot the absolute error between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.
    title : str, optional
        The title of the plot.
    cmap : str, optional
        The colormap to use, by default 'viridis'.

    Returns
    -------
    matplotlib.pyplot.Figure
        The figure containing the plot.
    """
    error = np.abs(img1 - img2)
    fig = plot_image(error, title=title or "Error Map", cmap=cmap)
    return fig


def plot_comparison(original, reconstructed, titles=None):
    """
    Plot original and reconstructed images side by side.

    Parameters
    ----------
    original : numpy.ndarray
        The original image.
    reconstructed : numpy.ndarray
        The reconstructed image.
    titles : list of str, optional
        The titles for the original and reconstructed images.

    Returns
    -------
    matplotlib.pyplot.Figure
        The figure containing the plots.
    """
    if titles is None:
        titles = ["Original", "Reconstructed"]
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title(titles[0])
    axs[1].imshow(reconstructed, cmap='gray')
    axs[1].set_title(titles[1])
    
    return fig
