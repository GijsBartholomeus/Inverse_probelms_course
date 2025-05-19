""" In this script, we create a dataset of reconstructed phantom images using two algorithms:
 Algoritme A and Algoritme B."""
# build_physical_dataset.py
import os, sys
# Add the parent directory containing 'utils' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, torch, deepinv as dinv
from tomophantom import TomoP2D
from PIL import Image
from pathlib import Path
from utils.methods_iradon import Algoritme_A, Algoritme_B 
import matplotlib.pyplot as plt

# ---------- user knobs ----------
ANGLES   = 100          # # projections
GAIN     = 1/40         # Poisson gain
MODEL   = 12   # see Phantom2DLibrary.dat for full list
OUT      = Path("data")
device   = "cpu"        # "cuda" if you like
# --------------------------------


# Load the phantoms
phantoms = np.load(os.path.join(OUT, "phantoms.npy"))  # load the phantoms

img_width = phantoms[0].shape[1]  # width of the phantom

# deepinv forward operator + Poisson noise
physics = dinv.physics.Tomography(img_width=img_width, angles=ANGLES,
                                  device=device,
                                  noise_model=dinv.physics.PoissonNoise(gain=GAIN))  # :contentReference[oaicite:1]{index=1}

number_of_phantoms = 10
recs_A = []
recs_B = []

for i in range(number_of_phantoms):
    print(f"Processing phantom {i+1}/{number_of_phantoms}...")
    # Load in random phantom
    phantom = phantoms[i]  
    x = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).to(device)
    
    # Create sinogram
    y = physics(x)  

    rec_A = Algoritme_A(y, physics, x_gt=x)
    rec_B = Algoritme_B(y, physics, x_gt=x)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 4, 1)
    # plt.imshow(phantom, cmap='gray')
    # plt.title('Original Phantom')
    # plt.axis('off')
    # plt.subplot(1, 4, 2)
    # plt.imshow(y.squeeze().cpu().numpy(), cmap='gray')
    # plt.title('Sinogram')
    # plt.axis('off')
    # plt.subplot(1, 4, 3)
    # plt.imshow(rec_A.squeeze().cpu().numpy(), cmap='gray')
    # plt.title('Reconstructed Phantom (Algoritme A)')
    # plt.axis('off')
    # plt.subplot(1, 4, 4)
    # plt.imshow(rec_B.squeeze().cpu().numpy(), cmap='gray')
    # plt.title('Reconstructed Phantom (Algoritme B)')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # Collect reconstructions in lists
    recs_A.append(rec_A.cpu().numpy())
    recs_B.append(rec_B.cpu().numpy())

   
# After loop, save arrays to .npy files
np.save(OUT / "full_images_algoA.npy", np.array(recs_A))
np.save(OUT / "full_images_algoB.npy", np.array(recs_B))

print(f"Done and saved {number_of_phantoms} reconstructions for Algoritme A and B to {OUT}")
