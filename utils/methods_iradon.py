"""
This file contains the implementation of two algorithms for image reconstruction from sinograms.
Algoritme A uses a Tikhonov prior with L2 data fidelity, while Algoritme B uses a TV prior with Poisson likelihood.
"""
import torch, deepinv as dinv
from pathlib import Path
import matplotlib.pyplot as plt
from deepinv.optim import optim_builder
from deepinv.optim.data_fidelity import L2, PoissonLikelihood
from deepinv.optim.prior         import Tikhonov, TVPrior





def Algoritme_A(sinogram, physics, x_gt=None):
    """
    Algoritme A
    :param sinogram: sinogram
    :param physics: physics
    :param x_gt: ground truth
    :return: reconstruction
    """
    # Load the sinogram
    y = sinogram

    # Prior 
    prior = Tikhonov()

    # Select the data fidelity term
    data_fidelity = L2()

    # Specific parameters for restoration with the given prior (Note that these parameters have not been optimized here)
    params_algo = {"stepsize": 0.0001, "lambda": 1e1}

    # Logging parameters
    verbose = True

    # Parameters of the algorithm to solve the inverse problem
    early_stop = True  # Stop algorithm when convergence criteria is reached
    crit_conv = "cost"  # Convergence is reached when the difference of cost function between consecutive iterates is
    # smaller than thres_conv
    thres_conv = 1e-5 # was 1e-5
    backtracking = False  # use backtraking to automatically adjust the stepsize
    max_iter = 1000  # Maximum number of iterations

    # Main algorithm
    model = optim_builder(
        iteration="GD",
        prior=prior,
        g_first=False,
        data_fidelity=data_fidelity,
        params_algo=params_algo,
        early_stop=early_stop,
        max_iter=max_iter,
        crit_conv=crit_conv,
        thres_conv=thres_conv,
        backtracking=backtracking,
        verbose=verbose,
        custom_init=lambda y, physics: {  # initialization of the algorithm using fbp
            "est": (physics.A_dagger(y), physics.A_dagger(y))
        }
    )


    # x_A = model(y, physics,x_gt = x_gt, compute_metric = True)
    x_A, metrics = model(
        y, physics, x_gt=x_gt, compute_metrics=True
    )  # reconstruction with PGD algorithm

    return x_A


def Algoritme_B(sinogram, physics, x_gt=None):
    """
    Algoritme B
    :param sinogram: sinogram
    :param physics: physics
    :param x_gt: ground truth
    :return: reconstruction
    """
    # Load the sinogram
    y = sinogram

    GAIN        = 1/40          # same as you used to corrupt the data
    max_iter_B  = 500
    # Prior
    prior_B   = TVPrior()

    # Data fidelity term
    fidelityB = PoissonLikelihood(gain=GAIN,bkg=0.01, denormalize= True)

    # Specific parameters for restoration with the given prior (Note that these parameters have not been optimized here)
    params_algo = {"stepsize": 0.0001, "lambda": 1e1}

    # Logging parameters
    verbose = True

    # Parameters of the algorithm to solve the inverse problem
    early_stop = True  # Stop algorithm when convergence criteria is reached
    crit_conv = "cost"  # Convergence is reached when the difference of cost function between consecutive iterates is
    # smaller than thres_conv
    thres_conv = 1e-5 # was 1e-5
    backtracking = False  # use backtraking to automatically adjust the stepsize


    model_B = optim_builder(
        iteration="GD",
        prior=prior_B,
        data_fidelity=fidelityB,
        params_algo=params_algo,   
        max_iter=max_iter_B,
        early_stop=early_stop,
        crit_conv=crit_conv,
        thres_conv= thres_conv,
        verbose=True,
        custom_init=lambda y, physics: {
            "est": (physics.A_dagger(y), physics.A_dagger(y))
        }
    )


    # x_A = model(y, physics,x_gt = x_gt, compute_metric = True)
    x_B, metrics = model_B(
        y, physics, x_gt=x_gt, compute_metrics=True
    )  # reconstruction with PGD algorithm

    return x_B