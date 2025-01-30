#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

def error_summary(errors):
    err_min = np.min(errors)
    err_max = np.max(errors)
    err_mean = np.mean(errors)
    err_sd = np.std(errors)/np.sqrt(len(errors))
    err_q1, err_median, err_q3 = np.quantile(errors, [0.25,0.5,0.75])
    err_iqr = err_q3 - err_q1
    return np.array([err_mean, err_sd, err_median, err_iqr, err_min, err_max])

ex_name = "BM_gam=0.5_N=500_d=1_D=100"
folder = os.path.join("/home", "soham", "Simulations", "spectral-NN",ex_name)
print(folder)

f_err1 = open(os.path.join("/home", "soham", "Simulations", "spectral-NN",ex_name,"Err_mean.txt"),"w")
f_err2 = open(os.path.join("/home", "soham", "Simulations", "spectral-NN",ex_name,"Err_median_range.txt"),"w")

for wt_fun in ["Parzen"]: #["truncated","Bartlett","Parzen","Tukey_Hanning","quadratic_spectral"]:
    q = 20
    files = os.listdir(os.path.join(folder,wt_fun+"_q="+str(q),"Results_New"))
    files.sort()
    print(files)
    f_err1.write(wt_fun+"\n")
    f_err2.write(wt_fun+"\n")
    print(wt_fun)
    for file in files:
        with open(os.path.join(folder,wt_fun+"_q="+str(q),"Results_New",file)) as f:
            f_cont = f.readlines()
            errors_total = []
            errors_cospect = []
            repl = 1
            current_line = 5
            flag = current_line <= len(f_cont)
            while flag:
                repl_id = int(f_cont[current_line-5].strip("\n").strip(":").strip("Example"))
                if not repl_id == repl:
                    print("Some problem with the example id! Aborting ...")
                    flag = False
                else:
                    rel_err = f_cont[current_line-3].strip("\n").split(" ")[-1]
                    err_cospect = f_cont[current_line-2].strip("\n").split(" ")[3].strip(",")
                    tr_cospect = f_cont[current_line-2].strip("\n").split(" ")[6].strip(",")
                    #err_quadspect = f_cont[current_line-1].strip("\n").split(" ")[3].strip(",")
                    errors_total.append(float(rel_err)*100.)
                    errors_cospect.append(float(err_cospect)/float(tr_cospect)*100.)
                    current_line += 6
                    repl += 1
                    flag = current_line <= len(f_cont)
            errors_total = np.asarray(errors_total)
            errors_cospect = np.asarray(errors_cospect)
            method = file.strip(".txt").strip("spectral_")
            err_tot = error_summary(errors_total)
            err_cospect = error_summary(errors_cospect)
            print(method)
            print(np.round(err_tot,decimals=2))
            print(np.round(err_cospect,decimals=2))
            f_err1.write(method+"\n")
            f_err1.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n" .format(err_tot[0],err_tot[1],err_cospect[0],err_cospect[1]))
            f_err2.write(method+"\n")
            f_err2.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n" .format(err_tot[2],err_tot[3],err_cospect[2],err_cospect[3]))
            f_err2.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n" .format(err_tot[4],err_tot[5],err_cospect[4],err_cospect[5]))
    f_err1.write("\n")
    f_err2.write("\n")
f_err1.close()
f_err2.close()


