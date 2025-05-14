"""
Pipeline module for filtered back-projection reconstruction
"""
import numpy as np
from data.phantoms import generate_shepp_logan
from data.sinogram import forward_project
from filters.filters import get_filter
from interpolation.methods import get_interpolator


class FBPipeline:
    """
    Pipeline for Filtered Back-Projection (FBP) reconstruction.

    Attributes
    ----------
    size : int
        Size of the square image (pixels).
    angles : np.ndarray
        Projection angles in degrees.
    noise_std : float
        Standard deviation of Gaussian noise added to sinogram.
    filter_type : str
        Name of the frequency filter to apply (e.g. 'ramp', 'hamming').
    interp_method : str
        Interpolation method for re-gridding ('nearest', 'linear', 'spline').
    phantom : np.ndarray
        Ground-truth image.
    sinogram : np.ndarray
        Noise-free sinogram.
    noisy_sinogram : np.ndarray
        Sinogram with added noise.
    fft_sino : np.ndarray
        1D Fourier transform of projections.
    filtered_sino : np.ndarray
        Filtered projection spectra.
    F_cartesian : np.ndarray
        Regridded 2D frequency domain array.
    reconstruction : np.ndarray
        Final reconstructed image.
    """

    def __init__(self,
                 size: int = 256,
                 num_angles: int = 180,
                 noise_std: float = 0.0,
                 filter_type: str = 'ramp',
                 interp_method: str = 'linear'):
        self.size = size
        self.angles = np.linspace(0, 180, num_angles, endpoint=False)
        self.noise_std = noise_std
        self.filter_type = filter_type
        self.interp_method = interp_method

        # placeholders
        self.phantom = None
        self.sinogram = None
        self.noisy_sinogram = None
        self.fft_sino = None
        self.filtered_sino = None
        self.F_cartesian = None
        self.reconstruction = None

        # modules
        self.filter = get_filter(filter_type)
        self.interpolator = get_interpolator(interp_method)

    def load_phantom(self):
        "Load in the phantom image"
        self.phantom = generate_shepp_logan(self.size)

    def simulate_sinogram(self):
        "Generate the sinogram from the phantom and add noise."
        self.sinogram = forward_project(self.phantom, self.angles, noise =self.noise_std)

    def compute_fft(self):
        # FFT along detector axis (axis=0)
        self.fft_sino = np.fft.fftshift(np.fft.fft(self.noisy_sinogram, axis=0), axes=0)

    def apply_filter(self):
        # filter is designed to work on freq axis matching fft_sino shape
        self.filtered_sino = self.filter.apply(self.fft_sino)

    def regrid_frequency(self):
        # produce 2D Cartesian freq grid
        self.F_cartesian = self.interpolator.regrid(self.filtered_sino, self.angles)

    def inverse_fft(self):
        # inverse 2D FFT to reconstruct
        img = np.fft.ifft2(np.fft.ifftshift(self.F_cartesian))
        self.reconstruction = np.real(img)

    def normalize(self):
        rec = self.reconstruction
        self.reconstruction = (rec - rec.min()) / (rec.max() - rec.min())

    def run(self):
        self.load_phantom()
        self.simulate_sinogram()
        self.compute_fft()
        self.apply_filter()
        self.regrid_frequency()
        self.inverse_fft()
        self.normalize()
        return self.reconstruction

    def get_results(self) -> dict:
        return {
            'phantom': self.phantom,
            'sinogram': self.sinogram,
            'noisy_sinogram': self.noisy_sinogram,
            'fft_sino': self.fft_sino,
            'filtered_sino': self.filtered_sino,
            'F_cartesian': self.F_cartesian,
            'reconstruction': self.reconstruction,
            'angles': self.angles,
        }


if __name__ == "__main__":
    pipeline = FBPipeline(size = 100, angles = np.linspace(0, 180, 180, endpoint=False), noise_std = 0.01)
# Set the image size and angles for the pipeline