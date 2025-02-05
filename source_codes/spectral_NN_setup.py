"""
Parameters used to fit the CovNet models
"""

import numpy as np
import torch
import Important_functions as Ifn

#directory = "C:\\Users\\Soham\\Git\\spectral-NN\\Data\\" # should be the directory where the data are located
directory = "/home/soham/Git/spectral-NN/Data/"

act_fn = torch.nn.Sigmoid() # activation function
init = torch.nn.init.xavier_normal_ # initialization method
#wt_fn = Ifn.kern_truncated # weight function
#wt_fn = Ifn.kern_bartlett 
wt_fn = Ifn.kern_parzen
#wt_fn = Ifn.kern_tukey_hanning
#wt_fn = Ifn.kern_quadratic_spectral
loss_grid = 100
optimizer = torch.optim.Adam
lr = 0.01
epochs = 1000 # number of epochs
burn_in = 750 # burn-in period
interval = 1  # interval after which the best state will be checked 
replicates = range(25)
