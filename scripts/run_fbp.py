#!/usr/bin/env python
"""
Script to run FBP reconstruction on a phantom image.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import phantoms, sinogram
from reconstruction import fbp
from pipelines.pipeline import Pipeline
from evaluation import visualize


def main():
    """
    Main function to run the FBP reconstruction.
    """
    parser = argparse.ArgumentParser(description='Run FBP reconstruction.')
    parser.add_argument('--size', type=int, default=256, help='Size of the phantom image')
    parser.add_argument('--filter', type=str, default='ramp', help='Filter to use for FBP')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level to add to sinogram')
    parser.add_argument('--output', type=str, help='Output file path for the results')
    
    args = parser.parse_args()
    
    # Create a phantom
    phantom = phantoms.shepp_logan(args.size)
    
    # Create the reconstructor
    reconstructor = fbp.FBPReconstructor(filter_name=args.filter)
    
    # Create and run the pipeline
    pipeline = Pipeline(reconstructor=reconstructor)
    results = pipeline.run(phantom=phantom, noise_level=args.noise)
    
    # Visualize the results
    fig = visualize.plot_comparison(
        results['phantom'],
        results['reconstruction'],
        titles=['Original Phantom', f'FBP Reconstruction (PSNR: {results["metrics"]["psnr"]:.2f} dB)']
    )
    
    # Save or show the results
    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
