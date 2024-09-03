#!/usr/bin/env python
# coding: utf-8

import datagen_AR1 as dn
import numpy as np
np.random.seed(12345)

N = 256
N0 = 100
K = 100
d = 1
gam = 0.5
replicates = 25
method = lambda s,t: dn.BM(s,t)
#theta = np.pi/4
#O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
O = None

dn.datagen_AR1_simple(N,K,d,replicates,method,O,gam,N0)