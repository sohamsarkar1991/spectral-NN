#!/usr/bin/env python
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def mat_scale(mat):
    m = np.max(np.abs(mat))
    if m == 0.:
        return 0.000000001
    return m

#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

import spectral_NN_setup as setup

dirc = setup.directory

u = np.loadtxt(dirc+"locations.dat",dtype="float32")
if len(u.shape)==1:
    d = 1
else:
    d = u.shape[1]
del u

true_loc = np.loadtxt(dirc+"True_locations_grid.dat",dtype="float32")
if d == 1:
    true_loc = true_loc.reshape(-1,1)
D_tr = true_loc.shape[0]
        
true_thetas = np.loadtxt(dirc+"True_thetas_grid.dat",dtype="float32")
K_theta = len(true_thetas)
spect_tr = np.loadtxt(dirc+"True_spectrum_grid.dat",dtype="float32")

if not os.path.isdir("Plots"):
    os.mkdir("Plots")
    os.mkdir(os.path.join("Plots","True"))
else:
    if not os.path.isdir(os.path.join("Plots","True")):
        os.mkdir(os.path.join("Plots","True"))

for t, theta in enumerate(true_thetas):
    cospect_tr = spect_tr[t*D_tr:(t+1)*D_tr,:D_tr]
    quadspect_tr = spect_tr[t*D_tr:(t+1)*D_tr,D_tr:]
    theta = true_thetas[t]
    
    fig, ax = plt.subplots(figsize=(2.5,2.5),ncols=1)
    m_ = mat_scale(cospect_tr)
    ax.imshow(cospect_tr/m_, origin='lower', cmap='seismic',vmin=-1,vmax=1)
    ax.set_xticks(np.linspace(0,D_tr-1,3))
    ax.set_xticklabels([0,0.5,1])
    ax.set_yticks(np.linspace(0,D_tr-1,3))
    ax.set_yticklabels([0,0.5,1])
    ax.set_title("{:.3f}" .format(m_), fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join("Plots","True","True_cospect_"+str(t+1)+".pdf"),
                bbox_inches="tight",dpi=300)
    
    fig, ax = plt.subplots(figsize=(2.5,2.5),ncols=1)
    m_ = mat_scale(quadspect_tr)
    ax.imshow(quadspect_tr/m_, origin='lower', cmap='seismic',vmin=-1,vmax=1)
    ax.set_xticks(np.linspace(0,D_tr-1,3))
    ax.set_xticklabels([0,0.5,1])
    ax.set_yticks(np.linspace(0,D_tr-1,3))
    ax.set_yticklabels([0,0.5,1])
    ax.set_title("{:.3f}" .format(m_), fontsize=15)
    fig.savefig(os.path.join("Plots","True","True_quadspect_"+str(t+1)+".pdf"),
                bbox_inches="tight",dpi=300)
    plt.close("all")
    