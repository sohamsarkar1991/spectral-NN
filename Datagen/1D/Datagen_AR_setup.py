#!/usr/bin/env python
# coding: utf-8

import datagen_AR1 as dn
import numpy as np
np.random.seed(12345)

folder = "Data"

### For data generation ###
N = 256
N0 = 100
gr_size = 100
d = 1
gam = 0.5
replicates = 25
method = lambda s,t: dn.BM(s,t)
#theta = np.pi/4
#O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
O = None

### For true spectrum ###
K = 100
M = 1000

dn.datagen_AR1_simple(N,gr_size,d,replicates,method,O,gam,N0,folder)
dn.true_spectrum_AR1_simple(K,M,d,replicates,method,O,gam,folder)