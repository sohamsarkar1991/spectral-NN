#!/usr/bin/env python
# coding: utf-8

"""
Module to create Spectral Neural Network models.
##### NEED TO UPDATE
A model can be created by using
CovNetShallow(d,N,R,act_fn,init) - for shallow
or CovNetDeep(d,N,R,depth,n_nodes,act_fn,init) - for deep
or CovNetDeepShared(d,N,R,depth,act_fn,init)
N - Number of fields (integer)
d - dimension (integer)
R, depth (L), n_nodes(p1=...=pL) - network parameters (integer)
act_fn - activation function \sigma; needs to be an element from torch.nn activation function
init - initialization for the weights of the model; biases are initialized as zero.
#####
"""

import torch
from collections import OrderedDict
from operator import methodcaller

##### Shallow spectral-NN #####
class spectralNNShallow(torch.nn.Module):
    def __init__(self,N,d,M,L,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(spectralNNShallow, self).__init__()
        self.N = N
        self.L = L
        self.act_fn = act_fn
        self.weight = torch.empty([M,2*L+1,d],dtype=torch.float32,requires_grad=True) #weights of the shallow networks
        self.bias = torch.zeros([M,2*L+1,1],dtype=torch.float32,requires_grad=True) #biases of the shallow networks
        self.xi = torch.empty([M,N+2*L],dtype=torch.float32,requires_grad=True) #the multipliers xi_{m,h}
        init(self.weight)
        init(self.xi)
        self.params = list([self.weight, self.xi, self.bias])

    def first_step(self, u):
        return self.act_fn(torch.einsum("ijk,lk -> ijl", self.weight, u) + self.bias) #an object of size M x 2L+1 x D

    def iter_prod(self, i, G): ## iterated product with the coefficients in xi
        return torch.einsum("ij,ijk -> k", self.xi[:,i:(i+2*self.L+1)], G).reshape(1,-1)

    def forward(self, u):
        G = self.first_step(u)
        return torch.cat([model.iter_prod(i,G) for i in range(self.N)])

##### Deep spectral-NN #####
class spectralNNDeep(torch.nn.Module):
    def __init__(self,N,d,M,L,depth,width,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(spectralNNDeep, self).__init__()
        self.N = N
        self.L = L
        self.depth = depth
        self.act_fn = act_fn
        self.init = init
        self.weight0 = torch.empty([M,2*L+1,width,d],dtype=torch.float32,requires_grad=True) #weights for the first hidden layer
        self.bias0 = torch.zeros([M,2*L+1,width,1],dtype=torch.float32,requires_grad=True) #biases for the first hidden layer
        self.weight = torch.empty([depth-2,M,2*L+1,width,width],dtype=torch.float32,requires_grad=True) #weights for the other hidden layers
        self.bias = torch.zeros([depth-2,M,2*L+1,width,1],dtype=torch.float32,requires_grad=True) #biases for the other hidden layer
        self.weight_final = torch.empty([M,2*L+1,width],dtype=torch.float32,requires_grad=True) #weights for the output layer
        self.bias_final = torch.zeros([M,2*L+1,1],dtype=torch.float32,requires_grad=True) #biases for the output layer
        self.xi = torch.empty([M,N+2*L],dtype=torch.float32,requires_grad=True) #the multipliers xi_{m,h}
        init(self.weight0)
        init(self.weight)
        init(self.weight_final)
        init(self.xi)
        self.params = list([self.weight0,self.bias0,self.weight,self.bias,self.weight_final,self.bias_final,self.xi])
        
    def first_step(self, u):
        u1 = self.act_fn(torch.einsum("ijkl,ml -> ijkm", self.weight0, u) + self.bias0)
        for i in range(self.depth-2):
            u1 = self.act_fn(torch.einsum("ijkl, ijlm -> ijkm", self.weight[i], u1) + self.bias[i])
        return self.act_fn(torch.einsum("ijk,ijkm -> ijm", self.weight_final, u1) + self.bias_final) #an object of size M x 2L+1 x D

    def iter_prod(self, i, G): ## iterated product with the coefficients in xi
        return torch.einsum("ij,ijk -> k", self.xi[:,i:(i+2*self.L+1)], G).reshape(1,-1)

    def forward(self, u):
        G = self.first_step(u)
        return torch.cat([model.iter_prod(i,G) for i in range(self.N)])

        
##### Deepshared spectral-NN #####
class CovNetDeepShared(torch.nn.Module):
    def __init__(self,d,N,R,depth,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        '''if depth==1:
            print('For depth=1, use CovNetShallow to define the model.')'''
        super(CovNetDeepShared, self).__init__()
        self.init = init
        layers = [('layer1',torch.nn.Linear(d,R,bias=True)),
                  ('act1',act_fn)]
        for i in range(depth-1):
            layers.append(('layer'+str(i+2),torch.nn.Linear(R,R,bias=True)))
            layers.append(('act'+str(i+2),act_fn))
        self.initial_layers = torch.nn.Sequential(OrderedDict(layers))
        self.initial_layers.apply(self.par_init_sequential)
        self.final_layer = torch.nn.Linear(R,N,bias=False)
        self.par_init(self.final_layer)
        self.params = list(self.initial_layers.parameters()) + list(self.final_layer.parameters())
    def par_init(self,m):
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        self.init(m.weight)
        
    def par_init_sequential(self,m):
        if type(m) == torch.nn.Linear:
            self.par_init(m)
            
    def first_step(self,u):
        return self.initial_layers(u)
    
    def forward(self,u):
        return self.final_layer(self.first_step(u)).T

