#!/usr/bin/env python
# coding: utf-8

import os, sys, time, shutil
import Summarize

if __name__ == "__main__":
    if len(sys.argv) > 7:
        dim = str(sys.argv[1])
        idx = str(sys.argv[2])
        N = int(sys.argv[3])
        gr_size = int(sys.argv[4])
        cov_type = str(sys.argv[5])
        if cov_type.lower() == "matern":
            par = float(sys.argv[6])
        else:
            par = "None"
        gam = float(sys.argv[7])
    print("Summary:\n")
    Summarize.__main__(idx=idx)
    print(time.ctime())
    print("\n")
            
    if cov_type.lower() == "bm":
        folder = "BM"
    elif cov_type.lower() == "ibm":
        folder = "IBM"
    elif cov_type.lower() == "matern":
        folder = "Matern_"+str(par)
            
    if not os.path.isdir(os.path.join("Results", dim)):
        os.mkdir(os.path.join("Results", dim))
    if not os.path.isdir(os.path.join("Results", dim, folder)):
        os.mkdir(os.path.join("Results", dim, folder))
    if not os.path.isdir(os.path.join("Results", dim, folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size))):
        os.mkdir(os.path.join("Results", dim, folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size)))

    new_folder = os.path.join("Results", dim, folder, "gam="+str(gam)+"_N="+str(N)+"_K="+str(gr_size))
            
    if os.path.exists(os.path.join("Results", "empirical.txt")):
        shutil.move(os.path.join("Results", "empirical.txt"), os.path.join(new_folder, "empirical.txt"))
    if os.path.exists(os.path.join("Results", "empirical_issues.txt")):
        shutil.move(os.path.join("Results", "empirical_issues.txt"), os.path.join(new_folder, "empirical_issues.txt"))
    if os.path.exists(os.path.join("Results", "spectral_NN.txt")):
        shutil.move(os.path.join("Results", "spectral_NN.txt"), os.path.join(new_folder, "spectral_NN.txt"))
    if os.path.exists(os.path.join("Results", "spectral_NN_issues.txt")):
        shutil.move(os.path.join("Results", "spectral_NN_issues.txt"), os.path.join(new_folder, "spectral_NN_issues.txt"))


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
