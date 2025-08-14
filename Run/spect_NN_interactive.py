#!/usr/bin/env python
# coding: utf-8

import os, sys, time
import torch
import numpy as np
import matplotlib.pyplot as plt

#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

import SpectralNetworks as spectNN
import Important_functions as Ifn
import spectral_NN_setup as setup

err_file = os.path.join("/home", "soham", "Git", "spectral-NN", "Results","spectral_NN.txt")
dirc = os.path.join("/home", "soham", "Git", "spectral-NN", "Data/")
repl = input("Replicate number (integer):")
repl = int(repl)-1

method = "Deep"
M, L = 10, 10
depth, width = 4, 20
q = 20

print('Example'+str(repl+1)+':')
u = np.loadtxt(dirc+"locations.dat",dtype="float32")
if len(u.shape)==1:
    D, d = len(u), 1
    u = u.reshape(D,1)
else:
    D, d = u.shape
u = torch.from_numpy(u)
x = np.loadtxt(dirc+'Example'+str(repl+1)+'.dat',dtype='float32')
N = x.shape[0]
if x.shape[1] != D:
    exit('Data shape mismatch!! Aborting..')
print('N='+str(N)+', D='+str(D)+', d='+str(d))

x = torch.from_numpy(x)
x -= torch.mean(x,dim=0,keepdim=True)

model = spectNN.spectralNNDeep(N,d,M,L,depth,width,setup.act_fn,setup.init)
check_file = "Deep_"+str(q)+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".pt"
optimizer = setup.optimizer(model.params,lr=setup.lr)
loss = Ifn.loss_spectralNN(N, setup.wt_fn, grid_size=setup.loss_grid, q=q)

l_tr = []
fit_time = 0
cont = "yes"
epoch = 0
min_l = np.inf
rep = 0
while cont.lower() == "yes":
    start_time = time.time()
    l_tr_, epoch_ = Ifn.spectral_NN_optim_best(x,u,model,loss,optimizer,
                                  epochs=100,burn_in=0,interval=setup.interval,
                                  checkpoint_file=check_file)
    fit_time += time.time() - start_time
    l_tr += l_tr_
    if l_tr_[epoch_-1] < min_l:
        min_l = l_tr_[epoch_-1]
        epoch = rep*100 + epoch
    plt.plot(l_tr_)
    plt.show()
    print("Minimum training error: {:.4f} obtained at epoch {}" .format(l_tr_[epoch_-1],rep*100+epoch_))
    rep += 1
    cont = input("Continue? [Yes/No]")

spect_dens_est = Ifn.spectral_density_evaluation(model, q=q, wt_fn=setup.wt_fn)
theta_file = dirc+"True_thetas"+str(repl+1)+".dat"
loc_file = dirc+"True_locations"+str(repl+1)+".dat"
spect_file = dirc+"True_spectrum"+str(repl+1)+".dat"

start_time = time.time()
test_err,num,den,tr_cospect,tr_quadspect,err_cospect,err_quadspect = Ifn.spectral_error_computation(spect_dens_est,theta_file,loc_file,spect_file)
eval_time = time.time() - start_time

print("Relative test error: {:.2f}%" .format(test_err*100))
print("Cospectra: Error - {:.6f}, Actual - {:.6f}" .format(err_cospect,tr_cospect))
print("Quadspectra: Error - {:.6f}, Actual - {:.6f}" .format(err_quadspect,tr_quadspect))

f_err = open(err_file,"a")
f_err.write("Example{}:\n" .format(repl+1))
f_err.write("Fitting time - {:.10f} seconds. Evaluation time - {:.10f} seconds.\n" .format(fit_time,eval_time))
f_err.write("Relative test errors - {:.10f}\n" .format(test_err))
f_err.write("Cospectra: Error - {:.10f}, Actual - {:.10f}\n" .format(err_cospect,tr_cospect))
f_err.write("Quadspectra: Error - {:.10f}, Actual - {:.10f}\n\n" .format(err_quadspect,tr_quadspect))
f_err.close()

