def emp_spect_dens_multirun(q,replicates=range(25)):
    import os, sys, time
    import numpy as np
    import torch

    #sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
    sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))
    
    import Important_functions as Ifn
    import spectral_NN_setup as setup

    if not os.path.isdir("Results"):
        os.mkdir("Results")

    err_file = os.path.join("Results","empirical.txt")

    dirc = setup.directory
    
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
        x -= torch.mean(x,dim=0,keepdim=True)

        start_time = time.time()
        emp_spect_dens = Ifn.empirical_spectral_density(x, u, q=q, wt_fn=setup.wt_fn)
        fit_time = time.time() - start_time
        
        theta_file = dirc+"True_thetas"+str(repl+1)+".dat"
        loc_file = dirc+"True_locations"+str(repl+1)+".dat"
        spect_file = dirc+"True_spectrum"+str(repl+1)+".dat"

        start_time = time.time()
        test_err,num,den,tr_cospect,tr_quadspect,err_cospect,err_quadspect = Ifn.emp_spectral_error_computation(emp_spect_dens,theta_file,loc_file,spect_file)
        eval_time = time.time() - start_time

        print("Relative test error: {:.2f}%" .format(test_err*100))

        f_err = open(err_file,"a")
        f_err.write("Example{}:\n" .format(repl+1))
        f_err.write("Fitting time - {:.10f} seconds. Evaluation time - {:.10f} seconds.\n" .format(fit_time,eval_time))
        f_err.write("Relative test error - {:.10f}\n" .format(test_err))
        f_err.write("Cospectra: Error - {:.10f}, Actual - {:.10f}\n" .format(err_cospect,tr_cospect))
        f_err.write("Quadspectra: Error - {:.10f}, Actual - {:.10f}\n\n" .format(err_quadspect,tr_quadspect))
        f_err.close()
 
    return 0.
