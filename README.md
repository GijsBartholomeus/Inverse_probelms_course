# Inverse_probelms_course
A project about the reconstruction of medical images.

## Project Structure

```
project_root/
│
├── data/
│   ├── __init__.py
│   ├── phantoms.py           # functions to generate or load test images
│   └── sinogram.py           # functions to forward-project an image
│
├── reconstruction/
│   ├── __init__.py
│   ├── fbp.py                # FBPReconstructor class
│   ├── idose.py              # iDoseReconstructor stub
│   └── imr.py                # IMRReconstructor stub
│
├── filters/
│   ├── __init__.py
│   └── frequency.py          # ramp, Shepp–Logan, Hamming, etc.
│
├── interpolation/
│   ├── __init__.py
│   └── methods.py            # nearest, linear, spline, custom…
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # MSE, PSNR, SSIM…
│   └── visualize.py          # plots, error maps
│
├── pipelines/
│   ├── __init__.py
│   └── pipeline.py           # Pipeline class orchestrating everything
│
├── notebooks/                # for ad-hoc exploration & plots
│
├── scripts/                  # CLI entry-points (e.g. run_fbp.py)
│
└── requirements.txt
```
