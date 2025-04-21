#!/usr/bin/env python
# coding: utf-8

import data_generation_functions as dn
import numpy as np
np.random.seed(54321)

folder = "Data" #where the data will be stored

### For data generation ###
N = 500 #number of curves/surfaces/fields
gr_size = 100 #number of grid points at which data will be generated
d = 2 #dimension
gam = 0 #autoregression coefficient
replicates = 25 #number of replicates
N0 = 100 #number of burn-in sample to achieve stationarity
method = lambda s,t: dn.BM(s,t) #specification of covariance kernel
#method = lambda s,t: dn.matern(s,t,nu=0.01) #specification of covariance kernel
#theta = np.pi/4
#O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
O = None #rotation of the covariance kernel

### For true spectrum ###
K = 100 # number of angles (theta)
M1 = 10000 # number of locations at which the true spectrum is evaluated (at each angle)
#M2 = 500 # resolution of the true grided spectrum (at each angle)

dn.datagen_AR1_simple(N,gr_size,d,replicates,method,O,gam,N0,folder)
dn.true_spectrum_AR1_simple(K,M1,d,replicates,method,O,gam,folder)
#dn.true_spectrum_grid_AR1_simple(K,M2,d,method,O,gam,folder)
