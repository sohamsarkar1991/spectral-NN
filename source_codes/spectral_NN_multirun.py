def spectral_NN_multirun(method,M,L,depth,width,q,replicates=range(25)):
    import os, sys, time
    import numpy as np
    import torch
    
    #sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
    sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))
    
    import SpectralNetworks as spectNN
    import Important_functions as Ifn
    import spectral_NN_setup as setup

    if not os.path.isdir("Results"):
        os.mkdir("Results")

    err_file = os.path.join("Results","spectral_NN.txt")

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


        if method.lower()=="shallow":
            model = spectNN.spectralNNShallow(N,d,M,L,setup.act_fn,setup.init)
            check_file = "Shallow_"+str(q)+"_"+str(M)+"_"+str(L)+".pt"
        elif method.lower()=="deep":
            model = spectNN.spectralNNDeep(N,d,M,L,depth,width,setup.act_fn,setup.init)
            check_file = "Deep_"+str(q)+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".pt"
        elif method.lower()=="deepshared1":
            model = spectNN.spectralNNDeepshared1(N,d,M,L,depth,width,setup.act_fn,setup.init)
            check_file = "Deepshared1_"+str(q)+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".pt"
        elif method.lower()=="deepshared2":
            model = spectNN.spectralNNDeepshared2(N,d,M,L,depth,width,setup.act_fn,setup.init)
            check_file = "Deepshared2_"+str(q)+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".pt"
        elif method.lower()=="deepshared3":
            model = spectNN.spectralNNDeepshared3(N,d,M,L,depth,width,setup.act_fn,setup.init)
            check_file = "Deepshared3_"+str(q)+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".pt"
        else:
            exit("Undefined model specified! Aborting...")

        optimizer = setup.optimizer(model.params,lr=setup.lr)
        loss = Ifn.loss_spectralNN(N, setup.wt_fn, grid_size=setup.loss_grid, q=q)
        print("Fitting the model ...")
        start_time = time.time()
        #l_tr = Ifn.spectral_NN_optim(x,u,model,loss,optimizer,epochs=setup.epochs,checkpoint_file=check_file)
        l_tr, epoch = Ifn.spectral_NN_optim_best(x,u,model,loss,optimizer,
                                          epochs=setup.epochs,burn_in=setup.burn_in,interval=setup.interval,
                                          checkpoint_file=check_file)
        fit_time = time.time() - start_time
        print("Model fitted. Time taken: {:.2f} seconds. Minimum loss achieved at epoch {}" .format(fit_time,epoch))

        spect_dens_est = Ifn.spectral_density_evaluation(model, q=q, wt_fn=setup.wt_fn)
        theta_file = dirc+"True_thetas"+str(repl+1)+".dat"
        loc_file = dirc+"True_locations"+str(repl+1)+".dat"
        spect_file = dirc+"True_spectrum"+str(repl+1)+".dat"

        start_time = time.time()
        test_err,num,den,tr_cospect,tr_quadspect,err_cospect,err_quadspect = Ifn.spectral_error_computation(spect_dens_est,theta_file,loc_file,spect_file)
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
