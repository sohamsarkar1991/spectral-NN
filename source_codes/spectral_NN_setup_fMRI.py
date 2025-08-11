"""
Parameters used to fit the spectral-NN models for fMRI data
"""

import numpy as np
import torch
import Important_functions as Ifn

act_fn = torch.nn.Sigmoid() # activation function
init = torch.nn.init.xavier_normal_ # initialization method
#wt_fn = Ifn.kern_truncated # weight function
#wt_fn = Ifn.kern_bartlett 
wt_fn = Ifn.kern_parzen
#wt_fn = Ifn.kern_tukey_hanning
#wt_fn = Ifn.kern_quadratic_spectral
optimizer = torch.optim.Adam
loss_grid = 100
lr = 0.01
epochs = 1000 # number of epochs
burn_in = 750 # burn-in period
interval = 1  # interval after which the best state will be checked 

