#!/usr/bin/env python
# coding: utf-8

import sys, os, shutil

if __name__ == "__main__":
    if len(sys.argv) > 6:
        dim = sys.argv[1]
        N = int(sys.argv[2])
        gr_size = int(sys.argv[3])
        cov_type = str(sys.argv[4])
        if cov_type.lower() == "matern":
            par = float(sys.argv[5])
        else:
            par = "None"
        gam = float(sys.argv[6])
        if dim=="1D":
            sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "Datagen", "1D"))
        elif dim=="2D":
            sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "Datagen", "2D"))
        elif dim=="3D":
            sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "Datagen", "3D"))
        from Datagen_AR import datagen
        datagen(cov=cov_type, par=par, replicates=25, N=N, gr_size=gr_size, gam=gam, true_spect_grid=False)
    else:
        print("Usage: python datagen.py <d> <N> <grid_size> <cov_type> <par> <gam>")

