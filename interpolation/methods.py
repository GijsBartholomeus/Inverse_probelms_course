"""
Interpolation methods for re-gridding polar frequency samples to Cartesian grid.
"""
import numpy as np
from scipy.interpolate import griddata


def get_interpolator(name: str):
    """
    Factory to get an interpolator by name.

    Supported methods:
      - 'nearest'
      - 'linear'
      - 'cubic'

    Returns an object with a `regrid(polar_data, angles)` method.
    """
    name = name.lower()
    if name == 'nearest':
        return NearestInterpolator()
    if name == 'linear':
        return LinearInterpolator()
    if name == 'cubic':
        return CubicInterpolator()
    raise ValueError(f"Unknown interpolation method '{name}'")


class BaseInterpolator:
    def regrid(self, polar_data: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Map polar frequency-domain data to a Cartesian grid.

        Parameters
        ----------
        polar_data : np.ndarray
            2D array of shape (num_freq, num_angles).
        angles : np.ndarray
            1D array of projection angles in degrees.

        Returns
        -------
        cart_data : np.ndarray
            2D Cartesian frequency-domain array of shape (num_freq, num_freq).
        """
        raise NotImplementedError


class NearestInterpolator(BaseInterpolator):
    def regrid(self, polar_data: np.ndarray, angles: np.ndarray) -> np.ndarray:
        return _griddata_interpolate(polar_data, angles, method='nearest')


class LinearInterpolator(BaseInterpolator):
    def regrid(self, polar_data: np.ndarray, angles: np.ndarray) -> np.ndarray:
        return _griddata_interpolate(polar_data, angles, method='linear')


class CubicInterpolator(BaseInterpolator):
    def regrid(self, polar_data: np.ndarray, angles: np.ndarray) -> np.ndarray:
        return _griddata_interpolate(polar_data, angles, method='cubic')


def _griddata_interpolate(polar_data: np.ndarray, angles: np.ndarray, method: str) -> np.ndarray:
    # dimensions
    num_freq, num_angles = polar_data.shape
    # normalized frequency coordinate
    freqs = np.linspace(-0.5, 0.5, num_freq, endpoint=False)

    # create polar sample points
    # for each angle index j and frequency index i, coord = (fx, fy)
    theta = np.deg2rad(angles)
    # meshgrid of freqs x angles
    fr, th = np.meshgrid(freqs, theta, indexing='ij')
    fx = fr * np.cos(th)
    fy = fr * np.sin(th)

    # prepare points and values for interpolation
    points = np.vstack((fx.ravel(), fy.ravel())).T
    values = polar_data.ravel()

    # create Cartesian grid (same resolution)
    grid_x = freqs
    grid_y = freqs
    gx, gy = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_points = np.vstack((gx.ravel(), gy.ravel())).T

    # perform interpolation
    cart_values = griddata(points, values, grid_points, method=method, fill_value=0)
    cart_data = cart_values.reshape((num_freq, num_freq))
    return cart_data
