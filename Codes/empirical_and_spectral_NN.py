#!/usr/bin/env python
# coding: utf-8

import os, sys, time, shutil

folder = os.path.join("/home", "soham", "Git", "spectral-NN", "Data")
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

##sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
##sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "Datagen", "1D"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(2, os.path.join("/home", "soham", "Git", "spectral-NN", "Datagen", "1D"))

from Datagen_AR import datagen
from spectral_NN_multirun import spectral_NN_multirun as spect_multirun
from empirical_spectral_density_multirun import emp_spect_dens_multirun as emp_multirun
import Summarize

par = None
gam = 0.5
#N = 500
gr_size = 200
for method in [("BM",None),("IBM",None),("Matern",0.001),("Matern",0.01),("Matern",0.1),("Matern",1)]:
    cov_type = method[0]
    par = method[1]
    #for gr_size in [20,40,80,160,320,640,1280,2560]:
    for N in [100,200,400,800,1600]:
    #for gam in [0.1,0.25,0.5,0.75,0.9]:
        print(time.ctime())
        print("Data generation:\n")
        datagen(cov=cov_type, par=par, replicates=25, N=N, gr_size=gr_size, gam=gam, true_spect_grid=False)
        print(time.ctime())
        print("\n")
    
        print("Spectral-NN model:\n")
        spect_multirun("Deep",M=10,L=10,depth=4,width=20,q=20)
        print(time.ctime())
        print("\n")
    
        print("Empirical spectral density:\n")
        emp_multirun(q=20)
        print(time.ctime())
        print("\n")
    
        print("Summary:\n")
        Summarize.__main__(idx="N")
        print(time.ctime())
        print("\n")
        
        if cov_type.lower() == "bm":
            folder = "BM"
        elif cov_type.lower() == "ibm":
            folder = "IBM"
        elif cov_type.lower() == "matern":
            folder = "Matern_"+str(par)
        
        if not os.path.isdir(os.path.join("Results", folder)):
            os.mkdir(os.path.join("Results", folder))
        if not os.path.isdir(os.path.join("Results", folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size))):
            os.mkdir(os.path.join("Results", folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size)))
        
        new_folder = os.path.join("Results", folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size))
        
        shutil.move(os.path.join("Results", "empirical.txt"), os.path.join(new_folder, "empirical.txt"))
        shutil.move(os.path.join("Results", "spectral_NN.txt"), os.path.join(new_folder, "spectral_NN.txt"))
