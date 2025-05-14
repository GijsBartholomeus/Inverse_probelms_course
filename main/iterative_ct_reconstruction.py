"""
iterative_ct_reconstruction.py  ✨ bug‑fix
========================================
A minimal but complete teaching implementation of three reconstruction
algorithms used throughout the Inverse Problems course.  This revision fixes a
**broadcasting error** in `huber_grad`/`tv_grad` that appeared on NumPy ≥ 1.26
and caused `ValueError: operands could not be broadcast together`.

Algorithms
----------
1. **Filtered Back‑Projection (FBP)** – analytical baseline.
2. **PWLS‑Huber** – hybrid IR approximating Philips *iDose⁴*.
3. **MBIR‑TV** – model‑based IR approximating Philips *IMR*.

The file contains no outside dependencies beyond *NumPy*, *SciPy* (for
`iradon`) and *scikit‑image* (for `radon/iradon`).
"""

from __future__ import annotations

import numpy as np
from math import sqrt
from typing import Optional
from skimage.transform import radon, iradon

try:
    from tqdm import trange
except ImportError:  # fallback if tqdm missing
    def trange(*args, **kwargs):  # type: ignore
        return range(*args)

###############################################################################
# Projector wrappers – keep A and Aᵀ abstract
###############################################################################

def forward_projection(x: np.ndarray, theta: np.ndarray, *, circle: bool = False) -> np.ndarray:
    """Radon transform wrapper (parallel‑beam)."""
    return radon(x, theta=theta, circle=circle)


def back_projection(sino: np.ndarray, theta: np.ndarray, shape: tuple[int, int], *, filter_name: Optional[str] = None) -> np.ndarray:
    """Adjoint of Radon (FBP if `filter_name` given)."""
    return iradon(sino, theta=theta, filter_name=filter_name, circle=False, output_size=shape[0])

###############################################################################
# Regulariser helpers – fixed broadcasting bug
###############################################################################

def _split_gradients(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward finite differences with Neumann boundary (copy last row/col)."""
    # dx = np.diff(x, axis=1, append=x[:, -1:])   # shape = H×W
    # dy = np.diff(x, axis=0, append=x[-1:, :])   # shape = H×W
    dx, dy =  np.gradient(x)
    return dx, dy


def huber_grad(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Huber gradient with Neumann boundary 0 (copy last row/col).
    Huber loss is a piecewise linear function that is quadratic for small values
    and linear for large values. It is used to reduce the influence of outliers
    in the data. The Huber gradient is defined as:
    g(x) = { x, |x| <= delta
             { delta * sign(x), |x| > delta

    where delta is a threshold parameter that determines the transition point
    """	
    dx, dy = _split_gradients(x)
    grad = np.zeros_like(x)

    # x‑direction contribution --------------------------------------------------
    mask = np.abs(dx) <= delta
    g_x = dx * mask + delta * np.sign(dx) * (~mask) 
    grad[:, :-1] += g_x[:, :-1]               # skip last col (ghost diff)

    # y‑direction contribution --------------------------------------------------
    mask = np.abs(dy) <= delta
    g_y = dy * mask + delta * np.sign(dy) * (~mask)
    grad[:-1, :] += g_y[:-1, :]               # skip last row

    return grad


def cauchy_grad(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Cauchy gradient with Neumann boundary 0 (copy last row/col).
    Cauchy loss is a robust loss function that is less sensitive to outliers
    than the squared loss. The Cauchy gradient is defined as:
    g(x) = x / (1 + (x / delta)^2)

    where delta is a threshold parameter that determines the transition point.
    """
    dx, dy = _split_gradients(x)
    mag = np.sqrt(dx**2 + dy**2 + 1e-8)
    grad = np.zeros_like(x)

    # x‑direction contribution --------------------------------------------------
    grad[:, :-1] += dx[:, :-1] / (1 + (dx[:, :-1] / delta)**2)  # skip last col

    # y‑direction contribution --------------------------------------------------
    grad[:-1, :] += dy[:-1, :] / (1 + (dy[:-1, :] / delta)**2)  # skip last row

    return grad


def tv_grad(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dx, dy = _split_gradients(x)
    mag = np.sqrt(dx**2 + dy**2 + eps**2)

    grad = np.zeros_like(x)
    grad[:, :-1] += dx[:, :-1] / mag[:, :-1]
    grad[:-1, :] += dy[:-1, :] / mag[:-1, :]
    return grad

###############################################################################
# Base reconstructor – dimension‑robust initialisation
###############################################################################

class IterativeReconstructor:
    def __init__(self, theta: np.ndarray, *, circle: bool = False):
        self.theta = np.asarray(theta, dtype=float)
        self.circle = circle
        self._shape: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    def reconstruct(self, sino: np.ndarray, *, n_iter: int, beta: float, **kw) -> np.ndarray:  # noqa: D401
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _init_image(self, sino: np.ndarray) -> np.ndarray:
        if self._shape is None:
            if self.circle:
                n = sino.shape[0]
            else:
                n = int(np.floor(sino.shape[0] / sqrt(2)))
            self._shape = (n, n)
        return back_projection(sino, self.theta, self._shape, filter_name="ramp")

    # Aliases for forward/adjoint ------------------------------------------------
    def _A(self, x: np.ndarray) -> np.ndarray:
        return forward_projection(x, self.theta, circle=self.circle)

    def _AT(self, y: np.ndarray) -> np.ndarray:
        return back_projection(y, self.theta, self._shape, filter_name=None)

###############################################################################
# 1) PWLS‑Huber (iDose‑like)
###############################################################################

class iDose(IterativeReconstructor):
    def reconstruct(self,sino: np.ndarray,n_iter: int = 30,beta: float = 0.01,delta: float = 1.0,step_size: float = 1e-2,save_steps: int = 5, prior: str = "huber") -> list[np.ndarray]:
        x = self._init_image(sino)  # Initialize with FBP
        # x = np.zeros_like(x)        # Initialize with zeros
        W = 1.0 / np.maximum(sino, 1.0)
        snapshots = []
        save_every = max(1, n_iter // save_steps)
        print(f"Saving every {save_every} iterations")
        for i in trange(n_iter, desc="IDose"):
            resid = self._A(x) - sino
            if prior == "huber":
                prior_grad = huber_grad(x, delta)
            elif prior == "cauchy":
                prior_grad = cauchy_grad(x, delta)
            else:
                raise ValueError(f"Unknown prior: {prior}")
            print(prior_grad)
            grad = self._AT(W * resid) - beta * prior_grad

            print('grad', grad)
            x -= step_size * grad
            if (i + 1) % save_every == 0 or (i + 1) == n_iter:
                snapshots.append(np.clip(x.copy(), 0.0, None))
        return snapshots

###############################################################################
# 2) MBIR‑TV (IMR‑like)
###############################################################################

class IMR(IterativeReconstructor):
    def reconstruct(self, sino: np.ndarray, *, n_iter: int = 60, beta: float = 0.002, L0: float = 1.0) -> np.ndarray:
        x = self._init_image(sino) # Initialize with FBP
        y, t = x.copy(), 1.0
        L = L0
        W = 1.0 / np.maximum(sino, 1.0)

        def grad(u: np.ndarray) -> np.ndarray:
            return self._AT(W * (self._A(u) - sino)) + beta * tv_grad(u)

        for _ in trange(n_iter, desc="MBIR‑FISTA"):
            g = grad(y)
            # Backtracking line‑search ------------------------------
            found = False
            while not found:
                z = y - g / L
                if self._sufficient_decrease(y, z, sino, W, beta, L):
                    found = True
                else:
                    L *= 2.0
            # FISTA momentum ---------------------------------------
            x_new = z
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            x, t = x_new, t_new
        return np.clip(x, 0.0, None)

    # helper ---------------------------------------------------------------
    def _objective(self, x: np.ndarray, sino: np.ndarray, W: np.ndarray, beta: float) -> float:
        resid = self._A(x) - sino
        return 0.5 * np.sum(W * resid**2) + beta * np.sum(np.sqrt(np.sum(np.stack(np.gradient(x))**2, axis=0) + 1e-8))

    def _sufficient_decrease(self, y: np.ndarray, z: np.ndarray, sino: np.ndarray, W: np.ndarray, beta: float, L: float) -> bool:
        lhs = self._objective(z, sino, W, beta)
        rhs = self._objective(y, sino, W, beta) + np.sum((z - y) * self._AT(W * (self._A(y) - sino))) + (L / 2) * np.sum((z - y)**2)
        return lhs <= rhs + 1e-5

###############################################################################
# Demo ------------------------------------------------------------------------
###############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    N = 256
    img = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True)
    theta = np.linspace(0, 180, max(img.shape), endpoint=False)
    sino = forward_projection(img, theta) 
    sino += 0.3 * np.random.randn(*sino.shape)  # Add noise

    fbp = back_projection(sino, theta, img.shape, filter_name="ramp")
    idose = iDose(theta).reconstruct(sino, n_iter=10, beta=10 **5, delta=0.0001, step_size=1e-2, save_steps=5, prior="cauchy")
    # mbir = MBIRReconstructor(theta).reconstruct(sino, n_iter=60, beta=0.002)


    fig2, axes2 = plt.subplots(1, len(idose), figsize=(4 * len(idose), 4))
    for i, (ax, im) in enumerate(zip(axes2, idose)):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"idose iter {((i+1)*len(idose))//len(idose)}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, title in zip(ax, [img, fbp, idose], ["Ground truth", "FBP", "idose "]):
        a.imshow(im, cmap="gray")
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    plt.show()
