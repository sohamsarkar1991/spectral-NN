"""
Parameters used to fit the CovNet models
"""

import numpy as np
import torch
import Important_functions as Ifn

#directory = "C:\\Soham\\Git\\spectral-NN\\Data\\" # should be the directory where the data are located
directory = "C:\\Users\\Soham\\Git\\spectral-NN\\Data\\"
#directory = "/home/soham/GitHub/spectral-NN/Data/"

act_fn = torch.nn.Sigmoid() # activation function
init = torch.nn.init.xavier_normal_ # initialization method
wt_fn = Ifn.kern_truncated # weight function
#wt_fn = Ifn.kern_bartlett 
#wt_fn = Ifn.kern_parzen
loss_grid = 100
optimizer = torch.optim.Adam
lr = 0.01
epochs = 1000 # number of epochs
replicates = range(25)
