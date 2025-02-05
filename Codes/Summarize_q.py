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

wt_fun = "Parzen"

f_err1 = open(os.path.join("/home", "soham", "Simulations", "spectral-NN",ex_name,wt_fun,"Err_mean.txt"),"w")
f_err2 = open(os.path.join("/home", "soham", "Simulations", "spectral-NN",ex_name,wt_fun,"Err_median_range.txt"),"w")

files = os.listdir(os.path.join(folder,wt_fun,"Results"))
files.sort()

qs = list(set([file.strip(".txt").split("_")[2] for file in files]))
qs.sort()

for q in qs:
    print(q)
    f_err1.write("q = "+str(q)+"\n")
    f_err2.write("q = "+str(q)+"\n")
    files_q = []
    for file in files:
        if file.strip(".txt").split("_")[2] == q:
            files_q.append(file)

    for file in files_q:
        with open(os.path.join(folder,wt_fun,"Results",file)) as f:
            method = file.strip(".txt").split("_")
            del method[0]
            del method[1]
            method = "_".join(method)
            print(method)
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
                    errors_total.append(float(rel_err)*100.)
                    errors_cospect.append(float(err_cospect)/float(tr_cospect)*100.)
                    current_line += 6
                    repl += 1
                    flag = current_line <= len(f_cont)
            errors_total = np.asarray(errors_total)
            errors_cospect = np.asarray(errors_cospect)
            err_tot = error_summary(errors_total)
            err_cospect = error_summary(errors_cospect)
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


