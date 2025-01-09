#!/usr/bin/env python
# coding: utf-8

#############################################################
##### Codes for generation of 1D functional time series #####
############ To be used in the case of d=1 only #############
#############################################################

import os
import numpy as np
from scipy import special
import itertools


def locations(K,d):
    gr = np.arange(1,K+1)/(K+1)
    grid = list(itertools.product(gr,repeat=d))
    grid = np.asarray(grid)
    return grid

def locations_unif(M,d):
    loc = np.random.uniform(low=0.,high=1.,size=[M,d])
    return loc

def pixelate_single(s,K):
    for j in np.arange(1,K):
        if s<(j+0.5)/(K+1):
            return j-1
    return K-1

def pixelate(u,K):
    idx = []
    for s in u:
        idx.append(pixelate_single(s,K))
    return idx

def indices(u,K,d):
    idx = np.apply_along_axis(pixelate,1,u,K)
    c = np.zeros(idx.shape[0],dtype=int)
    for i in range(d):
        c += idx[:,i]*K**(d-1-i)
    return c

def print_indices(u,v,K,d,file='indices.dat'):
    idx1 = indices(u,K,d)
    idx2 = indices(v,K,d)
    idx = np.vstack((idx1,idx2))
    np.savetxt(file,idx,fmt='%d')

def print_locations(K,d,file = 'locations.dat'):
    u = locations(K,d)
    np.savetxt(file,u,fmt='%.10f')
    
def bessel(s,t,nu):
    r = abs(s-t)
    c = 2**nu * special.gamma(nu+1) * r**(-nu) * special.jv(nu,r)
    return c

def BM(s,t):
    if s<=t and s>0.:
        return s
    elif s>t and t>0.:
        return t
    elif s<0. and t<0.:
        return BM(-s,-t)
    else:
        return 0.

def fBM(s,t,alpha):
    c = 0.5 * (abs(t)**(2*alpha) + abs(s)**(2*alpha) - abs(t-s)**(2*alpha))
    return c

def fBM2(s,t,H):
    alpha = 2*H
    if s<=t and s>0.:
        return 0.5*(t**alpha + s**alpha - (t-s)**alpha)
    elif s>t and t>0.:
        return 0.5*(t**alpha + s**alpha - (s-t)**alpha)
    elif s<0. and t<0.:
        return fBM2(-s,-t,H)
    else:
        return 0.

def iBM(s,t):
    if s<=t and s>0.:
        return s**2*t/2 - s**3/6
    elif s>t and t>0.:
        return t**2*s/2 - t**3/6
    elif s<0. and t<0.:
        return iBM(-s,-t)
    else:
        return 0.

def Bbridge(s,t,T=1.):
    if s<=t and s>0.:
        return s*(T-t)/T
    elif s>t and t>0.:
        return t*(T-s)/T
    elif s<0. and t<0.:
        return Bbridge(-s,-t,T)
    else:
        return 0.

def cauchy(s,t,gamma):
    r = abs(s-t)
    return 1./(1+r**2)**gamma

def dampedcos(s,t,lam):
    r = abs(s-t)
    c = np.exp(-lam*r) * np.cos(r)
    return c

def fractgauss(s,t,alpha):
    r = abs(s-t)
    c = 0.5*((r+1)**alpha + abs(r-1)**alpha - 2*r**alpha)
    return c

def matern(s,t,nu,rho=1.):
    r = abs(s-t)/rho
    if r==0:
        r = np.finfo(float).eps
    if nu == 0.5:
        c = np.exp(-r)
    elif nu == 1.5:
        c = r * np.sqrt(3)
        c = (1.+c) * np.exp(-c)
    elif nu == 2.5:
        c = r*np.sqrt(5)
        c = (1.+c+c**2/3.) * np.exp(-c)
    elif nu == np.inf:
        c = np.exp(-r**2/2.)
    else:  # general case; expensive to evaluate
        tmp = np.sqrt(2*nu) * r
        c = 2**(1.-nu) / special.gamma(nu)
        c *= tmp ** nu
        c *= special.kv(nu,tmp)
    return c

def cov_mat(K,d,method,O=None):
    u = locations(K,d)
    if O is not None:
        u = np.matmul(u,O.T)
    D = int(K**d)
    C = np.zeros([D,D],dtype=float)
    for i in range(D):
        s = u[i]
        for j in range(i,D):
            t = u[j]
            C[i,j] = method(s,t)
            C[j,i] = C[i,j]
    return C

def cross_cov(u,v,method,O=None):
    if u.shape != v.shape:
        print('Mismatch in shapes of u and v!')
        return
    M = u.shape[0]
    if O is not None:
        u = np.matmul(u,O.T)
        v = np.matmul(v,O.T)
    C = np.zeros(M,dtype=float)
    for i in range(M):
        s = u[i]
        t = v[i]
        C[i] = method(s,t)
    return C

def sqrt_mat(C):
    lam, E = np.linalg.eigh(C)
    idx = lam>0.
    lam = np.sqrt(lam[idx])
    E = E[:,idx]
    return [lam,E]


def datagen_AR1_simple(N,gr_size,d,replicates=1,method=None,rot=None,gam=0.5,N0=100,folder=""):
    """
    Data generation from an auto-regressive functional time series.
    INPUT:
        N - number of observations to be generated
        gr_size - grid size in 1D
        d - dimension
        replicates - number of replications
        method - covariance of the innovation process
        rot - rotation matrix to be applied
        gam - coefficient of the autoregression model
        N0 - burn-in sample: number of samples to be discarded for stationarity
    OUTPUT:
        Generates a simple AR(1) process X_t = gam * X_{t-1} + Z_t on a regular grid.
        Writes the locations of the grid points on the file 'locations.dat'. 
        Generates N curves at the grid points and writes them on
        'Examplex.dat', for x=1,...,replicates.
    """
    ##### For the innovation Z #####
    ##### Z and X_0 have the same distribution
    print("Data generation started ...")
    filename = os.path.join(folder,"locations.dat")
    print_locations(gr_size,d,filename)
        
    print('Basis generation started')
    lam, E = sqrt_mat(cov_mat(gr_size,d,method,rot))
    print('Basis generated')
    
    for repl in range(replicates):
        print('Replicate '+str(repl+1))
        filename = os.path.join(folder,"Example"+str(repl+1)+".dat")
        f=open(filename,'w')
        f.close()
        x0 = np.random.normal(loc=0.,scale=1,size=len(lam))
        x0 = np.sum(E*lam*x0,axis=1).reshape(1,-1)
        for n in range(N0): #burn-in, not to be saved
            z = np.random.normal(loc=0.,scale=1.,size=len(lam))
            z = np.sum(E*lam*z,axis=1).reshape(1,-1)
            x0 = gam*x0 + z
        f = open(filename,'a')
        for n in range(N): #after burn-in, to be saved
            z = np.random.normal(loc=0.,scale=1.,size=len(lam))
            z = np.sum(E*lam*z,axis=1).reshape(1,-1)
            x0 = gam*x0 + z
            np.savetxt(f,x0,fmt='%.10f')
        f.close()
    print("Data generation complete.")

def true_spectrum_AR1_simple(K,M,d,replicates=1,method=None,rot=None,gam=0.5,folder=""):
    """
    True spectrum of an autoregressive functional time series.
    INPUT:
        K - number of angles (theta)
        M - number of locations per angle/number of grid points
        d - dimension
        replicates - number of replications
        method - covariance of the innovation process
        rot - rotation matrix to be applied
        gam - coefficient of the autoregression model
    OUTPUT:
        Evaluates the spectrum of the model and writes
        them on "True_spectrumi.dat", for i=1,...,replicates.
        The angles are written on "True_thetasi.dat"
        and the locations are written on "True_locationsi.dat".
    """
    print("Writing true specturm ...")
    for repl in range(replicates):
        print("Replicate "+str(repl+1))
        thetas = np.random.uniform(low=-np.pi,high=np.pi,size=K)
        np.savetxt(os.path.join(folder,"True_thetas"+str(repl+1)+".dat"), thetas, fmt="%.10f")
        loc_file = os.path.join(folder,"True_locations"+str(repl+1)+".dat")
        spect_file = os.path.join(folder,"True_spectrum"+str(repl+1)+".dat")
        f_loc = open(loc_file,"w")
        f_loc.close()
        f_spect = open(spect_file,"w")
        f_spect.close()
        for theta in thetas:
            u = locations_unif(M,d)
            v = locations_unif(M,d)
            f_loc = open(loc_file,"a")
            np.savetxt(f_loc, np.hstack((u,v)), fmt="%.10f")
            f_loc.close()
            c_z = cross_cov(u,v,method,rot).reshape(-1,1)
            mult = 1./(1. + gam**2 - 2*gam*np.cos(theta))
            mult = mult/(2*np.pi)
            f_spect = open(spect_file,"a")
            np.savetxt(f_spect, np.hstack((mult*c_z,0*c_z)), fmt="%.10f")
            f_spect.close()
            del u,v,c_z,mult
    print("True spectrum writing complete.")

def true_spectrum_grid_AR1_simple(K,M,d,method=None,rot=None,gam=0.5,folder=""):
    """
    True spectrum of an autoregressive functional time series.
    INPUT:
        K - number of angles (theta)
        M - number of locations per angle/number of grid points
        d - dimension
        method - covariance of the innovation process
        rot - rotation matrix to be applied
        gam - coefficient of the autoregression model
    OUTPUT:
        Evaluates the spectrum of the model and writes
        them on "True_spectrum_grid.dat".
        The angles are written on "True_thetas_grid.dat"
        and the locations are written on "True_locations_grid.dat".
    """
    print("Writing true specturm ...")
    thetas = np.arange(start=-K,stop=K+0.5,step=1,dtype="float32")/K*np.pi
    np.savetxt(os.path.join(folder,"True_thetas_grid.dat"), thetas, fmt="%.10f")
    print_locations(M,d, file=os.path.join(folder,"True_locations_grid.dat"))
    spect_file = os.path.join(folder,"True_spectrum_grid.dat")
    f_spect = open(spect_file,"w")
    f_spect.close()
    c_z = cov_mat(M,d,method,rot)
    for theta in thetas:
        mult = 1./(1. + gam**2 - 2*gam*np.cos(theta))
        mult = mult/(2*np.pi)
        f_spect = open(spect_file,"a")
        np.savetxt(f_spect, np.hstack((mult*c_z,0.*c_z)), fmt="%.10f")
        f_spect.close()
    del c_z
    print("True spectrum writing complete.")
