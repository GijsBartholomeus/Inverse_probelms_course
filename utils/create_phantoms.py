"""A simple phantom generator for 2D tomography.
Idea comes form type 12 of the TomoP2D library dat file.


# import tomophantom
# lib2d = os.path.join(os.path.dirname(tomophantom.__file__),
#                      "phantomlib", "Phantom2DLibrary.dat")  
"""
import numpy as np
from skimage.draw import disk, polygon
import matplotlib.pyplot as plt

def make_phantom(sz=256,
                 n_disks=20,
                 n_segments=16,
                 r_core=0.36,      # inner-disk radius  (fraction of sz)
                 r_ring=0.4,       # ring-segment radius
                 seg_len=0.12,     # segment length     (fraction of sz)
                 seg_w=0.03,       # segment width      (fraction of sz)
                 seed=None):
    """
    Square phantom (float32):
      • random disks within a central circle (radius = r_core*sz)
      • n_segments rectangular bars on a larger ring (radius = r_ring*sz)
    Intensities: background=0, disks∈[0.3,0.9], ring=1.0
    """
    rng  = np.random.default_rng(seed)
    img  = np.zeros((sz, sz), np.float32)
    cx   = cy = sz / 2
    R0   = r_core*sz
    R1   = r_ring*sz

    # ---- disks --------------------------------------------------------
    centers = []                         # keep (x, y, radius) of accepted disks
    for _ in range(n_disks):
        rad = rng.uniform(0.02*sz, 0.08*sz)
        for _ in range(1000):          # try up to 1000 times
            dx, dy = rng.uniform(-R0+rad, R0-rad, 2)
            if dx*dx + dy*dy > (R0-rad)**2:   # outside core → reject
                continue
            # overlap-test
            if all((dx-xc)**2 + (dy-yc)**2 >= (rad+rc)**2
                for xc, yc, rc in centers):
                centers.append((dx, dy, rad))
                rr, cc = disk((cy+dy, cx+dx), rad, shape=img.shape)
                img[rr, cc] = rng.uniform(0.3, 0.9)
                break                     # go to next disk


    # ---- ring segments -----------------------------------------------
    hl, hw = seg_len*sz/2, seg_w*sz/2
    rect   = np.array([[-hl,-hw],[hl,-hw],[hl,hw],[-hl,hw]])     # axis-aligned

    for k in range(n_segments):
        theta = 2*np.pi*k / n_segments                 # ← rotate bars
        c, s  = np.cos(theta), np.sin(theta)
        R     = np.array([[-s,c],[c,s]])
        center = np.array([cx + R1*c, cy + R1*s])
        verts  = rect @ R.T + center
        rr, cc = polygon(verts[:,1], verts[:,0], img.shape)
        img[rr, cc] = 1.0

    return img


N        = 100                     # how many phantoms you want
phantoms = np.array([make_phantom() for _ in range(N)])

# save phantoms
np.save("data/phantoms.npy", phantoms)


# quick visual sanity-check
plt.imshow(phantoms[0], cmap='gray'); plt.axis('off'); plt.show()