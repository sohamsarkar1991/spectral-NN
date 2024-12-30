def spectral_NN_multirun(method,M,L,depth,width,q):
    import os
    import sys
    import time
    import torch
    import numpy as np
    #sys.path.insert(1, os.path.join("C:\\", "Soham", "Git", "spectral-NN", "source_codes"))
    sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
    #sys.path.insert(1, os.path.join("/home", "soham", "GitHub", "spectral-NN", "source_codes"))

    import SpectralNetworks as spectNN
    import Important_functions as Ifn
    import spectral_NN_setup as setup

    if not os.path.isdir("Results"):
        os.mkdir("Results")

    if method.lower()=="shallow":
        err_file = os.path.join("Results","spectral_Shallow_"+str(M)+"_"+str(L)+"_"+str(width)+".txt")
    else:
        err_file = os.path.join("Results","spectral_"+method+"_"+str(M)+"_"+str(L)+"_"+str(depth)+"_"+str(width)+".txt")

    dirc = setup.directory
    replicates = setup.replicates
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
        x = x - torch.mean(x,dim=0,keepdim=True)

        if method.lower()=="shallow":
            model = spectNN.spectralNNShallow(N,d,M,L,setup.act_fn,setup.init)
        elif method.lower()=="deep":
            model = spectNN.spectralNNDeep(N,d,M,L,depth,width,setup.act_fn,setup.init)
        elif method.lower()=="deepshared1":
            model = spectNN.spectralNNDeepshared1(N,d,M,L,depth,width,setup.act_fn,setup.init)
        elif method.lower()=="deepshared2":
            model = spectNN.spectralNNDeepshared2(N,d,M,L,depth,width,setup.act_fn,setup.init)
        elif method.lower()=="deepshared3":
            model = spectNN.spectralNNDeepshared3(N,d,M,L,depth,width,setup.act_fn,setup.init)
        else:
            exit("Undefined model specified! Aborting...")

        optimizer = setup.optimizer(model.params,lr=setup.lr)
        loss = Ifn.loss_spectralNN(N, setup.wt_fn, grid_size=setup.loss_grid, q=q)
        print("Fitting the model ...")
        start_time = time.time()
        Ifn.spect_NN_optimizer(x,u,model,loss,optimizer,epochs=setup.epochs)
        time_ellapsed = time.time() - start_time
        print("Model fitted. Time taken: {} seconds" .format(time_ellapsed))

        with torch.no_grad():
            num = loss.loss_fn(x,model(u)).item()
            den = loss.loss_fn(x,0*x).item()
            train_err = num/den
        print("Relative training error: {:.2f}%" .format(train_err*100))
        print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

        spect_dens_est = Ifn.spectral_density_evaluation(model, q=q, wt_fn=setup.wt_fn)
        theta_file = dirc+"True_thetas"+str(repl+1)+".dat"
        loc_file = dirc+"True_locations"+str(repl+1)+".dat"
        spect_file = dirc+"True_spectrum"+str(repl+1)+".dat"

        test_err,num,den,tr_re,tr_im,err_re,err_im = Ifn.spectral_error_computation(spect_dens_est,theta_file,loc_file,spect_file)

        print("Relative test error: {:.2f}%" .format(test_err*100))
        print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))
        print("Real part: Error - {:.4f}, Actual - {:.4f}" .format(err_re,tr_re))
        print("Imaginary part: Error - {:.4f}, Actual - {:.4f}" .format(err_im,tr_im))
        f_err = open(err_file,"a")
        f_err.write("Example{}:\n" .format(repl+1))
        f_err.write("Fitting time - {:.10f} seconds\n" .format(time_ellapsed))
        f_err.write("Relative errors: Training - {:.10f}\tTest - {:.10f}\n" .format(train_err,test_err))
        f_err.write("Real part: Error - {:.4f}, Actual - {:.4f}\n" .format(err_re,tr_re))
        f_err.write("Imaginary part: Error - {:.4f}, Actual - {:.4f}\n\n" .format(err_im,tr_im))
        f_err.close()
    return 0.