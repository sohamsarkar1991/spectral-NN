#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

import torch
import numpy as np


sys.path.insert(1, os.path.join("C:\\", "Soham", "Git", "spectral-NN", "source_codes"))
#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
#sys.path.insert(1, os.path.join("/home", "soham", "GitHub", "spectral-NN", "source_codes"))

import SpectralNetworks as spectNN
import Important_functions as Ifn
#import Other_functions as Ofn

#import current_setup as setup

dirc = "C:\\Soham\\Git\\spectral-NN\\Data\\"
#dirc = "C:\\Users\\Soham\\Git\\spectral-NN\\Data\\"
#dirc = "/home/soham/GitHub/spectral-NN/Data/"
repl = 0

print('Example'+str(repl+1)+':')
file = dirc+'locations'+str(repl+1)+'.dat'
u = np.loadtxt(dirc+"locations.dat",dtype="float32")
if len(u.shape)==1:
    D, d = len(u), 1
    u = u.reshape(D,1)
else:
    D, d = u.shape
u = torch.from_numpy(u)
file = dirc+'Example'+str(repl+1)+'.dat'
x = np.loadtxt(file,dtype='float32')
N = x.shape[0]
if x.shape[1] != D:
    exit('Data shape mismatch!! Aborting..')
print('N='+str(N)+', D='+str(D)+', d='+str(d))

x = torch.from_numpy(x)
x = x - torch.mean(x,dim=0,keepdim=True)

M, L, depth, width = 10, 10, 3, 20
act_fn=torch.nn.Sigmoid()
init=torch.nn.init.xavier_normal_

wt_fn = lambda x: np.exp(-x**2)
loss = Ifn.loss_spectralNN(N, wt_fn, grid_size=100, q=10)
epochs = 2000

print("\nFitting the shallow model ...")
model = spectNN.spectralNNShallow(N,d,M,L,act_fn,init)
optimizer = torch.optim.Adam(model.params,lr=0.01)

start_time = time.time()
l_tr = []
for epoch in range(epochs):
    l = loss.loss_fn(x,model(u))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    l_tr.append(l.item())
time_ellapsed = time.time() - start_time
print("Time taken: {:.3f} seconds" .format(time_ellapsed))

with torch.no_grad():
    num = loss.loss_fn(x,model(u)).item()
    den = loss.loss_fn(x,0*x).item()
    print("Relative error: {:.2f}%" .format(num/den*100))
    print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

spect_dens_est = Ifn.spectral_density_evaluation(model, q=10, wt_fn=wt_fn)
thetas = (2*torch.rand(100)-1)*np.pi
err_re = 0.
err_im = 0.
with torch.no_grad():
    for theta in thetas:
        f_hat = spect_dens_est.evaluate(theta, u, u)
        err_re += torch.norm(f_hat[:,0])**2
        err_im += torch.norm(f_hat[:,1])**2
        del f_hat
print("Average errors: Real part - {:.4f}, Imaginary part - {:.4f}\n" .format(err_re/100,err_im/100))


print("\nFitting the deep model ...")
model = spectNN.spectralNNDeep(N,d,M,L,depth,width,act_fn,init)
optimizer = torch.optim.Adam(model.params,lr=0.01)

start_time = time.time()
l_tr = []
for epoch in range(epochs):
    l = loss.loss_fn(x,model(u))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    l_tr.append(l.item())
time_ellapsed = time.time() - start_time
print("Time taken: {:.3f} seconds" .format(time_ellapsed))

with torch.no_grad():
    num = loss.loss_fn(x,model(u)).item()
    den = loss.loss_fn(x,0*x).item()
    print("Relative error: {:.2f}%" .format(num/den*100))
    print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

spect_dens_est = Ifn.spectral_density_evaluation(model, q=10, wt_fn=wt_fn)
thetas = (2*torch.rand(100)-1)*np.pi
err_re = 0.
err_im = 0.
with torch.no_grad():
    for theta in thetas:
        f_hat = spect_dens_est.evaluate(theta, u, u)
        err_re += torch.norm(f_hat[:,0])**2
        err_im += torch.norm(f_hat[:,1])**2
        del f_hat
print("Average errors: Real part - {:.4f}, Imaginary part - {:.4f}\n" .format(err_re/100,err_im/100))


print("\nFitting the deepshared Type-1 model ...")
model = spectNN.spectralNNDeepshared1(N,d,M,L,depth,width,act_fn,init)
optimizer = torch.optim.Adam(model.params,lr=0.01)

start_time = time.time()
l_tr = []
for epoch in range(epochs):
    l = loss.loss_fn(x,model(u))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    l_tr.append(l.item())
time_ellapsed = time.time() - start_time
print("Time taken: {:.3f} seconds" .format(time_ellapsed))

with torch.no_grad():
    num = loss.loss_fn(x,model(u)).item()
    den = loss.loss_fn(x,0*x).item()
    print("Relative error: {:.2f}%" .format(num/den*100))
    print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

spect_dens_est = Ifn.spectral_density_evaluation(model, q=10, wt_fn=wt_fn)
thetas = (2*torch.rand(100)-1)*np.pi
err_re = 0.
err_im = 0.
with torch.no_grad():
    for theta in thetas:
        f_hat = spect_dens_est.evaluate(theta, u, u)
        err_re += torch.norm(f_hat[:,0])**2
        err_im += torch.norm(f_hat[:,1])**2
        del f_hat
print("Average errors: Real part - {:.4f}, Imaginary part - {:.4f}\n" .format(err_re/100,err_im/100))


print("\nFitting the deepshared Type-2 model ...")
model = spectNN.spectralNNDeepshared2(N,d,M,L,depth,width,act_fn,init)
optimizer = torch.optim.Adam(model.params,lr=0.01)

start_time = time.time()
l_tr = []
for epoch in range(epochs):
    l = loss.loss_fn(x,model(u))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    l_tr.append(l.item())
time_ellapsed = time.time() - start_time
print("Time taken: {:.3f} seconds" .format(time_ellapsed))

with torch.no_grad():
    num = loss.loss_fn(x,model(u)).item()
    den = loss.loss_fn(x,0*x).item()
    print("Relative error: {:.2f}%" .format(num/den*100))
    print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

spect_dens_est = Ifn.spectral_density_evaluation(model, q=10, wt_fn=wt_fn)
thetas = (2*torch.rand(100)-1)*np.pi
err_re = 0.
err_im = 0.
with torch.no_grad():
    for theta in thetas:
        f_hat = spect_dens_est.evaluate(theta, u, u)
        err_re += torch.norm(f_hat[:,0])**2
        err_im += torch.norm(f_hat[:,1])**2
        del f_hat
print("Average errors: Real part - {:.4f}, Imaginary part - {:.4f}\n" .format(err_re/100,err_im/100))


print("\nFitting the deepshared Type-3 model ...")
model = spectNN.spectralNNDeepshared3(N,d,M,L,depth,width,act_fn,init)
optimizer = torch.optim.Adam(model.params,lr=0.01)

start_time = time.time()
l_tr = []
for epoch in range(epochs):
    l = loss.loss_fn(x,model(u))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    l_tr.append(l.item())
time_ellapsed = time.time() - start_time
print("Time taken: {:.3f} seconds" .format(time_ellapsed))

with torch.no_grad():
    num = loss.loss_fn(x,model(u)).item()
    den = loss.loss_fn(x,0*x).item()
    print("Relative error: {:.2f}%" .format(num/den*100))
    print("Numerator: {:.4f}, Denominator: {:.4f}" .format(num,den))

spect_dens_est = Ifn.spectral_density_evaluation(model, q=10, wt_fn=wt_fn)
thetas = (2*torch.rand(100)-1)*np.pi
err_re = 0.
err_im = 0.
with torch.no_grad():
    for theta in thetas:
        f_hat = spect_dens_est.evaluate(theta, u, u)
        err_re += torch.norm(f_hat[:,0])**2
        err_im += torch.norm(f_hat[:,1])**2
        del f_hat
print("Average errors: Real part - {:.4f}, Imaginary part - {:.4f}\n" .format(err_re/100,err_im/100))