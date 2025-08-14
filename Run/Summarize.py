#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

def summary(values):
    na_ = sum(np.isnan(values))
    if na_ > 0:
        values = np.compress(~np.isnan(values),values)
        print("Non-numeric outcomes!!")
    mean = np.mean(values)
    sd = np.std(values)/np.sqrt(len(values))
    return np.asarray([mean, sd])

def __main__(idx=""):
    folder = os.path.join("/home", "soham", "Git", "spectral-NN", "Results")

    errs = np.empty([4], dtype=float)
    times = np.empty([8], dtype=float)
    err_file = open(os.path.join(folder,"Errors"+str(idx)+".txt"),"a")
    time_file = open(os.path.join(folder,"Computing_times"+str(idx)+".txt"),"a")

    for i,fname in enumerate(["empirical", "spectral_NN"]):
        file = os.path.join(folder,fname+".txt")
        if os.path.exists(file):
            with open(file) as f:
                f_cont = f.readlines()
                errors = []
                fit_times = []
                eval_times = []
                repl = 1
                current_line = 0
                flag = len(f_cont)>=current_line+5
                while flag:
                    repl_id = int(f_cont[current_line].strip("\n").strip(":").strip("Example"))
                    if not repl_id == repl:
                        print("Problem with the example id! Aborting ...")
                        flag = False
                    else:
                        fit_time = float(f_cont[current_line+1].strip("\n").split(" ")[3])
                        fit_times.append(fit_time)
                        eval_time = float(f_cont[current_line+1].strip("\n").split(" ")[8])
                        eval_times.append(eval_time)
                        err = float(f_cont[current_line+2].strip("\n").split(" ")[-1])*100
                        errors.append(err)
                        current_line += 6
                        repl += 1
                        flag = len(f_cont)>= current_line+5
                summ = np.round(summary(errors),2)
                errs[2*i] = summ[0]
                errs[2*i+1] = summ[1]
                summ = np.round(summary(fit_times),5)
                times[2*i] = summ[0]
                times[2*i+4] = summ[1]
                summ = np.round(summary(eval_times),5)
                times[2*i+1] = summ[0]
                times[2*i+5] = summ[1]
        else:
            errs[2*i] = 'nan'
            errs[2*i+1] = 'nan'
            times[2*i] = 'nan'
            times[2*i+4] = 'nan'
            times[2*i+1] = 'nan'
            times[2*i+5] = 'nan'
    
    print("Errors: empirical - {}% ({}), spectral-NN - {}% ({})".format(errs[0],errs[1],errs[2],errs[3]))
    print("Average time taken: empirical - {:.3f} + {:.3f} sec., spectral-NN - {:.3f} + {:.3f} sec." 
          .format(times[0],times[1],times[2],times[3]))
    
    np.savetxt(err_file, errs, fmt="%.2f", newline="\t")
    err_file.write("\n")
    err_file.close()
    
    np.savetxt(time_file, times, fmt="%.3f", newline="\t")
    time_file.write("\n")
    time_file.close()
    return 0.
