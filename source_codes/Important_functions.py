#!/usr/bin/env python
# coding: utf-8

##### Important functions for fitting the CovNet model #####


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

##### vectorization #####
def kern_truncated(hs):
    return np.array(list(map(truncated,hs))).astype("float32")

def kern_bartlett(hs):
    return np.array(list(map(bartlett,hs))).astype("float32")

def kern_parzen(hs):
    return np.array(list(map(parzen,hs))).astype("float32")

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
        self.thetas = torch.arange(start=-self.q/(self.q+1),end=self.q/(self.q+1),step=1/(self.q+1),dtype=torch.float32,requires_grad=False)*np.pi
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
        #I11 = torch.matmul(x,x.T)
        #I22 = torch.matmul(x_tilde,x_tilde.T)
        #I12 = torch.matmul(x,x_tilde.T)
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

        #for h1 in range(self.q):
        #    for h2 in range(self.q):
        #        A[self.q+h1,self.q+h2] = self.inner_sum(I11,I22,I12,h1,h2)
        #        A[self.q-h1,self.q-h2] = A[self.q+h1,self.q+h2]
        #        A[self.q-h1,self.q+h2] = A[self.q+h1,self.q+h2]
        #        A[self.q+h1,self.q-h2] = A[self.q+h1,self.q+h2]
        
        return A

    def loss_fn(self, x, x_tilde):
        A = self.inner_part(x, x_tilde)
        l = 0.
        for theta in self.thetas:
            #l += torch.sqrt(torch.matmul(self.w,torch.matmul(torch.cos(theta*self.C_diff)*A,self.w)))/(self.N)
            l += torch.sqrt(torch.einsum("i,ij,j ->", self.w,torch.cos(theta*self.C_diff)*A,self.w))/(self.N)
        return l/(2*self.q+1)

##### Module for evaluation of estimated spectral density #####

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
            auv = torch.einsum("ijklm, ikn, jln -> mn", self.lam, self.model.first_step(u), self.model.first_step(v)) # a_h(u,v) size q x D
            fuv_re = torch.einsum("h,h,hj -> j", self.w, torch.cos(self.hs*theta), auv).reshape(-1,1)
            fuv_im = torch.einsum("h,h,hj -> j", self.w, torch.sin(self.hs*theta), auv).reshape(-1,1)
            return torch.cat([fuv_re, fuv_im], dim=1)


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
    def __init__(self, filename='checkpoint_spectNN.pt'):
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

""" Need to modify """
def cnet_optim_best(x,u,model,loss_fn,optimizer,split,epochs=1000,burn_in=500,interval=1,checkpoint_file='Checkpoint.pt'):
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
