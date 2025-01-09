#!/usr/bin/env python
# coding: utf-8

##### Important functions for fitting the spectral-NN model #####


import torch
import numpy as np

##### Train-validation splitting #####
##### Using mini batches #####

def batch_CV(D,batch_size=10): # points randomly shuffled each time, iterated through all points
    if batch_size==None or batch_size>=D:
        yield np.arange(D)
    else:
        folds = int(np.ceil(D/batch_size))
        indices = np.random.choice(D,D,replace=False)
        fold = 0
        while fold<folds-1:
            yield np.sort(indices[fold*batch_size:(fold+1)*batch_size])
            fold += 1
        yield np.sort(indices[fold*batch_size:])

##### Kernel functions for covariance estimation #####

def truncated(s):
    if np.abs(s) <= 1.:
        return 1.
    else:
        return 0.

def bartlett(s):
    t = np.abs(s)
    if t <= 1.:
        return 1-t
    else:
        return 0.

def parzen(s):
    t = np.abs(s)
    if t < 0.5:
        return 1 - 6*(t**2) + 6*(t**3)
    elif t <= 1.:
        return 2*((1-t)**3)
    else:
        return 0.

def tukey_hanning(s):
    if np.abs(s) <= 1.:
        return (1+np.cos(np.pi*s))/2
    else:
        return 0.

def quadratic_spectral(s):
    t = 6*np.pi*s/5
    if np.abs(s) > 0.:
        return 3*(np.sin(t)/t - np.cos(t))/(t**2)
    else:
        return 0.

##### vectorization #####
def kern_truncated(hs):
    return np.array(list(map(truncated,hs))).astype("float32")

def kern_bartlett(hs):
    return np.array(list(map(bartlett,hs))).astype("float32")

def kern_parzen(hs):
    return np.array(list(map(parzen,hs))).astype("float32")

def kern_tukey_hanning(hs):
    return np.array(list(map(tukey_hanning,hs))).astype("float32")

def kern_quadratic_spectral(hs):
    return np.array(list(map(quadratic_spectral,hs))).astype("float32")

##### Loss module #####
class loss_spectralNN:
    """Module to compute the loss function associated with the spectral NN estimator"""
    def __init__(self, N, wt_fn, grid_size = 100, q=10):
        """
        Args:
            N - sample size (length of functional time-series)
            wt_fn - a function to compute the weights 
                        for spectral density estimation.
            grid_size - size of the discrete grid on [-pi,pi]
                        for choice of theta.
            q - lag value for spectral density estimation.
        """
        self.N = N
        self.q = q
        self.gr = int(grid_size/2)
        #self.thetas = torch.arange(start=-self.gr,end=self.gr+0.5,step=1.,dtype=torch.float32,requires_grad=False)/(self.gr+1)*np.pi
        hs = np.arange(start=-self.q,stop=self.q+0.5,step=1.,dtype="float32")
        self.C_diff = torch.from_numpy(np.array([[h1-h2 for h2 in hs] for h1 in hs]))
        self.w = torch.from_numpy(wt_fn(hs/self.q))
    

    def inner_sum(self,I11,I22,I12,h1,h2):
        """Function used to compute the elements of A"""
        """Only defined for non-negative h1,h2"""
        """
        Args:
            I11, I22, I12 - inner-product matrices.
            h1, h2 - lag indices
        """
        if h1 == 0 and h2 == 0:
            a1 = torch.sum(I11*I11)
            a2 = torch.sum(I22*I22)
            a3 = torch.sum(I12*I12)
            return a1 + a2 - 2*a3
        elif h1 == 0:
            a1 = torch.sum(I11[:,h2:]*I11[:,:-h2])
            a2 = torch.sum(I22[:,h2:]*I22[:,:-h2])
            a3 = torch.sum(I12[:,h2:]*I12[:,:-h2])
            a4 = torch.sum(I12[h2:,:]*I12[:-h2,:])
            return a1 + a2 - a3 - a4
        elif h2 == 0:
            a1 = torch.sum(I11[h1:,:]*I11[:-h1,:])
            a2 = torch.sum(I22[h1:,:]*I22[:-h1,:])
            a3 = torch.sum(I12[h1:,:]*I12[:-h1,:])
            a4 = torch.sum(I12[:,h1:]*I12[:,:-h1])
            return a1 + a2 - a3 - a4
        else:
            a1 = torch.sum(I11[h1:,h2:]*I11[:-h1,:-h2])
            a2 = torch.sum(I22[h1:,h2:]*I22[:-h1,:-h2])
            a3 = torch.sum(I12[h1:,h2:]*I12[:-h1,:-h2])
            a4 = torch.sum(I12[h2:,h1:]*I12[:-h2,:-h1])
            return a1 + a2 - a3 - a4

    def inner_part(self, x, x_tilde):
        """
        Calculates the inner part of the loss function a(h,h') for h,h'=-q,...,q

        Args:
            x - observed functional time series (NxD matrix)
            x_tilde - fitted time seris using neural networks (NxD matrix)
        """
        I11 = torch.einsum("ik,jk -> ij", x, x)
        I22 = torch.einsum("ik,jk -> ij", x_tilde, x_tilde)
        I12 = torch.einsum("ik,jk -> ij", x, x_tilde)
        A = torch.zeros([2*self.q+1,2*self.q+1],dtype=torch.float32,requires_grad=False)
        for h1 in range(self.q):
            for h2 in range(h1,self.q):
                A[self.q+h1,self.q+h2] = self.inner_sum(I11,I22,I12,h1,h2)
                A[self.q-h1,self.q-h2] = A[self.q+h1,self.q+h2]
                A[self.q-h1,self.q+h2] = A[self.q+h1,self.q+h2]
                A[self.q+h1,self.q-h2] = A[self.q+h1,self.q+h2]
                A[self.q+h2,self.q+h1] = A[self.q+h1,self.q+h2]
                A[self.q-h2,self.q-h1] = A[self.q+h1,self.q+h2]
                A[self.q-h2,self.q+h1] = A[self.q+h1,self.q+h2]
                A[self.q+h2,self.q-h1] = A[self.q+h1,self.q+h2]
        
        return A/(self.N**2)

    def loss_fn(self, x, x_tilde):
        A = self.inner_part(x, x_tilde)
        l = 0.
        self.thetas = np.random.uniform(low=-np.pi,high=np.pi,size=2*self.gr+1)
        for theta in self.thetas:
            l += torch.sqrt(torch.einsum("i,ij,j ->", self.w,torch.cos(theta*self.C_diff)*A,self.w))/(2*np.pi)
        return l

##### Module for evaluation of estimated spectral density #####

class empirical_spectral_density:
    """Module to evaluate the empirical spectral density"""
    def __init__(self, x, q, wt_fn):
        N, D = x.shape
        self.lagged_covariance = torch.zeros([2*q+1,D,D],dtype=torch.float32,requires_grad=False)
        for i in range(N):
            self.lagged_covariance[q,:,:] += torch.einsum("i,j -> ij", x[i,:], x[i,:])
        for h in range(1,q+1):
            for i in range(N-h):
                self.lagged_covariance[h,:,:] += torch.einsum("i,j -> ij", x[i,:], x[i+h,:])
            self.lagged_covariance[h+q,:,:] = self.lagged_covariance[h,:,:]
        self.lagged_covariance = self.lagged_covariance/N
        
        self.hs = np.arange(start=-q,stop=q+0.5,step=1.,dtype="float32")
        self.w = torch.from_numpy(wt_fn(self.hs/q))
        self.hs = torch.from_numpy(self.hs)

    def evaluate(self, theta):
        co_spect = torch.einsum("h,h,hij -> ij", self.w, torch.cos(self.hs*theta), self.lagged_covariance)/(2*np.pi)
        quad_spect = torch.einsum("h,h,hij -> ij", self.w, torch.sin(self.hs*theta), self.lagged_covariance)/(2*np.pi)
        return list([co_spect, quad_spect])


class spectral_density_evaluation:
    """Module to evaluate the spectral density estimator for a fitted spectral-NN model"""
    def __init__(self, model, q, wt_fn):
        self.N = model.N
        self.M = model.M
        self.L = model.L
        self.q = q
        self.model = model
        hs = np.arange(start=-self.q,stop=self.q+0.5,step=1.,dtype="float32")
        self.w = torch.from_numpy(wt_fn(hs/self.q))
        self.hs = torch.from_numpy(hs)

        self.lam = torch.empty([self.M,self.M,2*self.L+1,2*self.L+1,2*self.q+1],dtype=torch.float32,requires_grad=False)
        with torch.no_grad():
            for h in range(2*self.q+1):
                N1 = self.N-h
                for j1 in range(2*self.L+1):
                    for j2 in range(2*self.L+1):
                        self.lam[:,:,j1,j2,h] = torch.einsum("ij,kj -> ik", model.xi[:,(j1+h):(j1+h+N1)], model.xi[:,j2:(j2+N1)])/self.N

    def evaluate(self, theta, u, v):
        with torch.no_grad():
            #Gu = self.model.first_step(u) # ((g_{m,h}(u))) size M x 2L+1 x D
            #Gv = self.model.first_step(v) # ((g_{m,h}(v))) size M x 2L+1 x D
            ah_uv = torch.einsum("ijklm, ikn, jln -> mn", self.lam, self.model.first_step(u), self.model.first_step(v)) # a_h(u,v) size 2q+1 x D
            co_spect = (torch.einsum("h,h,hj -> j", self.w, torch.cos(self.hs*theta), ah_uv)/(2*np.pi)).reshape(-1,1)
            quad_spect = (torch.einsum("h,h,hj -> j", self.w, torch.sin(self.hs*theta), ah_uv)/(2*np.pi)).reshape(-1,1)
            return torch.cat([co_spect, quad_spect], dim=1)

    def evaluate_grid(self, theta, u):
        """ evaluation of fitted spectral density on a dense grid """
        with torch.no_grad():
            Gu = self.model.first_step(u) # ((g_{m,h}(u))) size M x 2L+1 x D
            ah_u = torch.einsum("ijklm, ikp, jlq -> mpq", self.lam, Gu, Gu) #a_h(u) size 2q+1 x D x D
            co_spect = (torch.einsum("h,h,hij -> ij", self.w, torch.cos(self.hs*theta), ah_u))/(2*np.pi)
            quad_spect = (torch.einsum("h,h,hij -> ij", self.w, torch.sin(self.hs*theta), ah_u))/(2*np.pi)
            return list([co_spect, quad_spect])


##### Early stopping routine #####
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=1e-4, filename='checkpoint_spectNN.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            filename : file on which the best model will be stored
                            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.epoch = None
        self.early_stop = False
        self.delta = delta
        self.checkpoint = filename

    def __call__(self, val_loss, model, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)
        elif score >= self.best_score or 1.-score/self.best_score < self.delta:
            #If validation error starts to increase or the decrease is less than the threshold
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_checkpoint(model)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.epoch = epoch
            self.counter = 0

    def save_checkpoint(self, model):
        '''
        Save model when validation loss increase or doesn't decrease more than a certain level
        '''
        torch.save(model.state_dict(), self.checkpoint)
        
    def load_checkpoint(self, model):
        '''
        Reset the model to the saved checkpoint
        '''
        model.load_state_dict(torch.load(self.checkpoint))
        
##### Save and load best state_dict #####        
class BestState:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filename="checkpoint_spectNN.pt"):
        """
        Args:
            filename : file on which the best model will be stored
                            
        """
        self.best_score = None
        self.file = filename
        self.epoch = None

    def __call__(self, error, model,epoch):

        score = error

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        '''
        Save model
        '''
        torch.save(model.state_dict(), self.file)
        
    def load_checkpoint(self, model):
        '''
        Reset the model to the saved checkpoint
        '''
        model.load_state_dict(torch.load(self.file))


##### Optimization routine #####

def spect_NN_optimizer(x,u,model,loss,optimizer,epochs=1000,checkpoint_file="checkpoint.pt"):
    l_tr = []
    #best_state = BestState(checkpoint_file)
    for epoch in range(epochs):
        l = loss.loss_fn(x,model(u))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_tr.append(l.item())
    torch.save(model.state_dict(), checkpoint_file)
    return l_tr
    #return 0.

""" Need to modify """
def spect_NN_optim_best(x,u,model,loss_fn,optimizer,split,epochs=1000,burn_in=500,interval=1,checkpoint_file='Checkpoint.pt'):
    """
    Optimization routine with on-the-go error computation. Returns the model state that produced the best error.
    INPUTS - x, u - data and locations
             model - the model to be fitted
             loss_fn - loss function (MSE/COV/COV2)
             optimizer - optimizer to be used. An element of class torch.optim
             split - training and validation splitter function
             epochs - number of epochs
             plot_filename - the filename in which the plots will be saved
             
    OUTPUTS - l_tr - training errors
              l_va - validation errors
    """
    D = u.shape[0]
    l_tr = []
    l_va = []
    
    best_state = BestState(checkpoint_file)
    
    for epoch in range(burn_in):
        for Q_tr in split(D):
            loss = loss_fn(x[:,Q_tr],model(u[Q_tr,:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    for epoch in range(burn_in,epochs):
        train_losses = []
        val_losses = []
        for Q_tr in split(D):
            loss = loss_fn(x[:,Q_tr],model(u[Q_tr,:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            #with torch.no_grad():
            #    loss = loss_fn(x[:,Q_va],model(u[Q_va,:]))
            #    val_losses.append(loss.item())
        l_tr.append(np.mean(train_losses))
        #l_va.append(np.mean(val_losses))
        with torch.no_grad():
            loss = loss_fn(x,model(u))
            l_va.append(loss.item())
        if (epoch-burn_in)%interval == interval-1:
            best_state(l_va[-1],model,epoch)

    best_state.load_checkpoint(model)
    epoch = best_state.epoch
    return l_tr, l_va, epoch+1


##### Error computation #####
def spectral_error_computation(spect_est,theta_file="True_thetas.dat",loc_file="True_locations.dat",spect_file="True_spectrum.dat"):
    tr_thetas = np.loadtxt(theta_file,dtype="float32")
    tr_loc = np.loadtxt(loc_file,dtype='float32')
    spect_tr = np.loadtxt(spect_file,dtype="float32")
    K = len(tr_thetas)
    D_star = int(tr_loc.shape[0]/K)
    d = int(tr_loc.shape[1]/2)
    if int(spect_tr.shape[0]/K) != D_star:
        exit('Shape mismatch!! Aborting..')
    tr_thetas = torch.from_numpy(tr_thetas)
    tr_loc = torch.from_numpy(tr_loc)
    spect_tr = torch.from_numpy(spect_tr)

    tr_cospect = 0.
    tr_quadspect = 0.
    err_cospect = 0.
    err_quadspect = 0.
    den = 0.
    num = 0.
    with torch.no_grad():
        for i,theta in enumerate(tr_thetas):
            u_tr = tr_loc[i*D_star:(i+1)*D_star,:d]
            v_tr = tr_loc[i*D_star:(i+1)*D_star,d:]
            f_hat = spect_est.evaluate(theta,u_tr,v_tr)
            f_tr = spect_tr[i*D_star:(i+1)*D_star,:]
            num_cospect = torch.norm(f_tr[:,0]-f_hat[:,0]).item()
            den_cospect = torch.norm(f_tr[:,0]).item()
            num_quadspect = torch.norm(f_tr[:,1]-f_hat[:,1]).item()
            den_quadspect = torch.norm(f_tr[:,1]).item()
            tr_cospect += den_cospect
            tr_quadspect += den_quadspect
            err_cospect += num_cospect
            err_quadspect += num_quadspect
            num += np.sqrt(num_cospect**2 + num_quadspect**2)
            den += np.sqrt(den_cospect**2 + den_quadspect**2)

    tr_cospect = tr_cospect/(K*np.sqrt(D_star))
    tr_quadspect = tr_quadspect/(K*np.sqrt(D_star))
    err_cospect = err_cospect/(K*np.sqrt(D_star))
    err_quadspect = err_quadspect/(K*np.sqrt(D_star))
    num = num/(K*np.sqrt(D_star))
    den = den/(K*np.sqrt(D_star))
    test_err = num/den

    return [test_err,num,den,tr_cospect,tr_quadspect,err_cospect,err_quadspect]
