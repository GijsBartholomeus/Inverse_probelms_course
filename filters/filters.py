"""
Frequency-domain filters for FBP reconstruction.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_filter(name: str):
    """
    Factory to get a frequency filter by name.

    Supported filters:
      - 'ramp'
      - 'shepp-logan'
      - 'hamming'

    Parameters
    ----------
    name : str
        Filter identifier.

    Returns
    -------
    filter_obj
        An object with an `apply(fft_sino: np.ndarray) -> np.ndarray` method.
    """
    name = name.lower()
    if name == 'ramp':
        return RampFilter()
    if name == 'shepp-logan':
        return SheppLoganFilter()
    if name == 'hamming':
        return HammingFilter()
    raise ValueError(f"Unknown filter '{name}'")


class BaseFilter:
    def apply(self, fft_sino: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RampFilter(BaseFilter):
    """
    Ramp filter: multiplies frequency spectrum by |f|.
    """
    def apply(self, fft_sino: np.ndarray) -> np.ndarray:
        # fft_sino shape: (num_freq, num_angles)
        num_freq = fft_sino.shape[0]

        # normalized frequency axis [-0.5, 0.5)
        freqs = np.linspace(-0.5, 0.5, num_freq, endpoint=False)
        ramp = np.abs(freqs)
        # apply to each projection for each angle
        # ramp[:, None] makes it a column vector for broadcasting
        return fft_sino * ramp[:, None]


class SheppLoganFilter(BaseFilter):
    """
    Shepp-Logan filter: ramp * sinc(f / (2*fc)), where fc=0.5
    """
    def apply(self, fft_sino: np.ndarray) -> np.ndarray:
        num_freq = fft_sino.shape[0]
        freqs = np.linspace(-0.5, 0.5, num_freq, endpoint=False)
        # avoid division by zero
        sinc = np.ones_like(freqs)
        mask = freqs != 0
        sinc[mask] = np.sin(np.pi * freqs[mask]) / (np.pi * freqs[mask])
        ramp = np.abs(freqs)
        filt = ramp * sinc
        return fft_sino * filt[:, None]


class HammingFilter(BaseFilter):
    """
    Hamming filter: ramp * hamming window.
    """
    def apply(self, fft_sino: np.ndarray) -> np.ndarray:
        num_freq = fft_sino.shape[0]
        freqs = np.linspace(-0.5, 0.5, num_freq, endpoint=False)
        # standard Hamming window over [-0.5,0.5]
        window = 0.54 + 0.46 * np.cos(2 * np.pi * freqs)
        ramp = np.abs(freqs)
        filt = ramp * window
        return fft_sino * filt[:, None]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example: plot filter responses
    num_freq = 512
    freqs = np.linspace(-0.5, 0.5, num_freq, endpoint=False)
    ones = np.ones((num_freq, 1))
    filters = {
        'ramp': RampFilter(),
        'shepp-logan': SheppLoganFilter(),
        'hamming': HammingFilter()
    }
    plt.figure()
    for name, filt in filters.items():
        response = filt.apply(ones)[:, 0]
        plt.plot(freqs, response, label=name)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    plt.title('Filter Responses')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Import own functions
    from data.phantoms import generate_shepp_logan
    from data.sinogram import forward_project

    # Load the Shepp-Logan phantom
    x = generate_shepp_logan(size = 216)
    
    # Create a sinogram from the Shepp-Logan phantom
    sino = forward_project(image = x, angles = np.linspace(0, 180, 180, endpoint=False), noise = 0.01)
    # Create a figure with subplots
    num_filters = len(filters)
    fig, axes = plt.subplots(1, num_filters + 1, figsize=(15, 5))

    # Plot the original sinogram on the left
    axes[0].imshow(np.abs(sino), aspect='auto', cmap='gray')
    axes[0].set_title('Original Sinogram')
    axes[0].set_xlabel('Angles')
    axes[0].set_ylabel('Frequency')

    # Apply each filter and plot the result
    for i, (name, filt) in enumerate(filters.items(), start=1):
        filtered_sino = filt.apply(sino)
        axes[i].imshow(np.abs(filtered_sino), aspect='auto', cmap='gray')
        axes[i].set_title(f'{name.capitalize()} Filtered')
        axes[i].set_xlabel('Angles')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()