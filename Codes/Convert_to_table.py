#!/usr/bin/env python
# coding: utf-8

import numpy as np

par = "Gr_size"
dim = 1

if par=="N":
    idx = [100,200,400,800,1600]
elif par=="AR_coef":
    idx = [0.1,0.25,0.5,0.75,0.9]
elif par=="Gr_size":
    if dim==1:
        idx = [20,40,80,160,320,640,1280,2560]
    elif dim==2:
        idx = [10,20,40,80,160,320]
    elif dim==3:
        idx = [5,10,20,40,80,160]

data = np.loadtxt("Errors"+par+".txt",dtype="float")

if data.shape[1]!=4 or data.shape[0]/6!=len(idx):
    exit("Something wrong with the error file! Aborting.")

emp_err = data[:,0]
emp_sd = data[:,1]
spectNN_err = data[:,2]
spectNN_sd = data[:,3]

f = open("Err_table.txt","w")
for i,j in enumerate(idx):
    f.write("{}" .format(j))
    for k in range(6):
        f.write("\t&& {:.2f}\t& {:.2f}" .format(emp_err[k*len(idx)+i],spectNN_err[k*len(idx)+i]))
    f.write("\\\\\n")
    f.write("")
    for k in range(6):
        f.write("\t&& \\se{{{:.2f}}}\t& \\se{{{:.2f}}}" .format(emp_sd[k*len(idx)+i],spectNN_sd[k*len(idx)+i]))
    f.write("\\\\ [2pt] \n")

f.close()

