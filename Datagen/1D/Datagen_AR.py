#!/usr/bin/env python
# coding: utf-8

import data_generation_functions as dn
import numpy as np

def datagen(cov="BM",par=None,replicates=25,N=200,gr_size=50,gam=0.5,true_spect_grid=False):
    """
    cov - covariance kernel. BM/IBM/Matern
    par - (optional) parameter for the kernel. Ignored for BM/IBM 
    replicates - number of replications
    N - number of curves/surfaces/fields
    gr_size - number of grid (per dimension) points at which data will be generated
    gam - autoregression coefficient
    true_spect_grid - (boolean) whether to evaluate the true spectrum on a grid
    """
    d = 1 #dimension
    #np.random.seed(54321)
    folder = "Data" #where the data will be stored

    ### For data generation ###
    N0 = 100 #number of burn-in samples to achieve stationarity
    O = None #no rotation of covariance
    if cov.lower()=="bm":
        method = lambda s,t: dn.BM(s,t)
    elif cov.lower()=="ibm":
        method = lambda s,t: dn.iBM(s,t)
    elif cov.lower()=="matern":
        method = lambda s,t: dn.matern(s,t,nu=par)
    else:
        exit("Unknown covariance specification! Aborting.")

    dn.datagen_AR1_simple(N,gr_size,d,replicates,method,O,gam,N0,folder)

    ### For true spectrum ###
    K = 100 # number of angles (theta)
    M1 = 10000 # number of locations at which the true spectrum is evaluated (at each angle)
    dn.true_spectrum_AR1_simple(K,M1,d,replicates,method,O,gam,folder)

    if true_spect_grid:
        M2 = 500 # resolution of the true grided spectrum (at each angle)
        dn.true_spectrum_grid_AR1_simple(K,M2,d,method,O,gam,folder)

    return 0.
