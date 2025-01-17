#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

import Important_functions as Ifn
import spectral_NN_setup as setup

def mat_scale(mat):
    m = np.max(np.abs(mat))
    if m == 0.:
        return 0.000000001
    return m

q = input("Bandwidth (q): ")
q = int(q)

dirc = setup.directory
replicates = setup.replicates

if not os.path.isdir("Plots"):
    os.mkdir("Plots")

for repl in replicates:
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
    x = x - torch.mean(x,dim=0,keepdim=True)

    if not os.path.isdir(os.path.join("Plots","Ex"+str(repl+1))):
        os.mkdir(os.path.join("Plots","Ex"+str(repl+1)))
        os.mkdir(os.path.join("Plots","Ex"+str(repl+1),"Empirical"))
    else:
        if not os.path.isdir(os.path.join("Plots","Ex"+str(repl+1),"Empirical")):
            os.mkdir(os.path.join("Plots","Ex"+str(repl+1),"Empirical"))

    true_loc = np.loadtxt(dirc+"True_locations_grid.dat",dtype="float32")
    if d == 1:
        true_loc = true_loc.reshape(-1,1)
    true_loc = torch.from_numpy(true_loc)

    emp_spect_dens = Ifn.empirical_spectral_density(x, q=q, wt_fn=setup.wt_fn)
    true_thetas = np.loadtxt(dirc+"True_thetas_grid.dat",dtype="float32")

    K_theta = len(true_thetas)
    spect_tr = np.loadtxt(dirc+"True_spectrum_grid.dat",dtype="float32")
    
    for t, theta in enumerate(true_thetas):
        theta = true_thetas[t]
        cospect_emp, quadspect_emp = emp_spect_dens.evaluate(theta)
    
        fig, ax = plt.subplots(figsize=(2.5,2.5),ncols=1)
        m_ = mat_scale(cospect_emp.numpy())
        ax.imshow(cospect_emp/m_, origin='lower', cmap='seismic',vmin=-1,vmax=1)
        ax.set_xticks(np.linspace(0,D-1,3))
        ax.set_xticklabels([0,0.5,1])
        ax.set_yticks(np.linspace(0,D-1,3))
        ax.set_yticklabels([0,0.5,1])
        ax.set_title("{:.3f}" .format(m_), fontsize=15)
        fig.savefig(os.path.join("Plots","Ex"+str(repl+1),"Empirical","Empirical_cospect_"+str(t+1)+".pdf"),
                    bbox_inches="tight",dpi=300)
    
        fig, ax = plt.subplots(figsize=(2.5,2.5),ncols=1)
        m_ = mat_scale(quadspect_emp.numpy())
        ax.imshow(quadspect_emp/m_, origin='lower', cmap='seismic',vmin=-1,vmax=1)
        ax.set_xticks(np.linspace(0,D-1,3))
        ax.set_xticklabels([0,0.5,1])
        ax.set_yticks(np.linspace(0,D-1,3))
        ax.set_yticklabels([0,0.5,1])
        ax.set_title("{:.3f}" .format(m_), fontsize=15)
        fig.savefig(os.path.join("Plots","Ex"+str(repl+1),"Empirical","Empirical_quadspect_"+str(t+1)+".pdf"),
                    bbox_inches="tight",dpi=300)
        plt.close("all")

