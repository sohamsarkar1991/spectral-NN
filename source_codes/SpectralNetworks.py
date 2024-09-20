#!/usr/bin/env python
# coding: utf-8

"""
Module to create Spectral Neural Network models.
A model can be created by using
spectralNNShallow(N,d,M,L,act_fn,init) - for shallow
or spectralNNDeep(N,d,M,L,depth,width,act_fn,init) - for deep
or spectralNNDeepshared(N,d,M,L,depth,width,act_fn,init) - for deepshared
N - Number of fields (integer)
d - dimension (integer)
M, L, depth, width - network parameters (integer)
act_fn - activation function \sigma; needs to be an element from torch.nn activation function
init - initialization for the weights of the model; biases are initialized as zero.
"""

import torch

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
        return torch.cat([self.iter_prod(i,G) for i in range(self.N)])

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
        return torch.cat([self.iter_prod(i,G) for i in range(self.N)])

        
##### Deepshared spectral-NN #####
class spectralNNDeepshared1(torch.nn.Module):
    ### Deepshared spectral-NN of the first type (weight sharing accross m and h)
    def __init__(self,N,d,M,L,depth,width,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(spectralNNDeepshared1, self).__init__()
        self.N = N
        self.L = L
        self.depth = depth
        self.act_fn = act_fn
        self.init = init
        self.weight0 = torch.empty([width,d],dtype=torch.float32,requires_grad=True) #weights for the first hidden layer
        self.bias0 = torch.zeros([width,1],dtype=torch.float32,requires_grad=True) #biases for the first hidden layer
        self.weight = torch.empty([depth-1,width,width],dtype=torch.float32,requires_grad=True) #weights for the other hidden layers
        self.bias = torch.zeros([depth-1,width,1],dtype=torch.float32,requires_grad=True) #biases for the other hidden layer
        self.weight_final = torch.empty([M,2*L+1,width],dtype=torch.float32,requires_grad=True) #weights for the output layer
        self.bias_final = torch.zeros([M,2*L+1,1],dtype=torch.float32,requires_grad=True) #biases for the output layer
        self.xi = torch.empty([M,N+2*L],dtype=torch.float32,requires_grad=True) #the multipliers xi_{m,h}
        init(self.weight0)
        init(self.weight)
        init(self.weight_final)
        init(self.xi)
        self.params = list([self.weight0,self.bias0,self.weight,self.bias,self.weight_final,self.bias_final,self.xi])
        
    def first_step(self, u):
        u1 = self.act_fn(torch.einsum("kl,ml -> km", self.weight0, u) + self.bias0)
        for i in range(self.depth-1):
            u1 = self.act_fn(torch.einsum("kl, lm -> km", self.weight[i], u1) + self.bias[i])
        return self.act_fn(torch.einsum("ijk,km -> ijm", self.weight_final, u1) + self.bias_final) #an object of size M x 2L+1 x D

    def iter_prod(self, i, G): ## iterated product with the coefficients in xi
        return torch.einsum("ij,ijk -> k", self.xi[:,i:(i+2*self.L+1)], G).reshape(1,-1)

    def forward(self, u):
        G = self.first_step(u)
        return torch.cat([self.iter_prod(i,G) for i in range(self.N)])


class spectralNNDeepshared2(torch.nn.Module):
    ### Deepshared spectral-NN of the second type (weight sharing accross h for each m)
    def __init__(self,N,d,M,L,depth,width,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(spectralNNDeepshared2, self).__init__()
        self.N = N
        self.L = L
        self.depth = depth
        self.act_fn = act_fn
        self.init = init
        self.weight0 = torch.empty([M,width,d],dtype=torch.float32,requires_grad=True) #weights for the first hidden layer
        self.bias0 = torch.zeros([M,width,1],dtype=torch.float32,requires_grad=True) #biases for the first hidden layer
        self.weight = torch.empty([depth-1,M,width,width],dtype=torch.float32,requires_grad=True) #weights for the other hidden layers
        self.bias = torch.zeros([depth-1,M,width,1],dtype=torch.float32,requires_grad=True) #biases for the other hidden layer
        self.weight_final = torch.empty([M,2*L+1,width],dtype=torch.float32,requires_grad=True) #weights for the output layer
        self.bias_final = torch.zeros([M,2*L+1,1],dtype=torch.float32,requires_grad=True) #biases for the output layer
        self.xi = torch.empty([M,N+2*L],dtype=torch.float32,requires_grad=True) #the multipliers xi_{m,h}
        init(self.weight0)
        init(self.weight)
        init(self.weight_final)
        init(self.xi)
        self.params = list([self.weight0,self.bias0,self.weight,self.bias,self.weight_final,self.bias_final,self.xi])

    def first_step(self, u):
        u1 = self.act_fn(torch.einsum("ikl,ml -> ikm", self.weight0, u) + self.bias0)
        for i in range(self.depth-1):
            u1 = self.act_fn(torch.einsum("ikl, ilm -> ikm", self.weight[i], u1) + self.bias[i])
        return self.act_fn(torch.einsum("ijk,ikm -> ijm", self.weight_final, u1) + self.bias_final) #an object of size M x 2L+1 x D

    def iter_prod(self, i, G): ## iterated product with the coefficients in xi
        return torch.einsum("ij,ijk -> k", self.xi[:,i:(i+2*self.L+1)], G).reshape(1,-1)

    def forward(self, u):
        G = self.first_step(u)
        return torch.cat([self.iter_prod(i,G) for i in range(self.N)])


class spectralNNDeepshared3(torch.nn.Module):
    ### Deepshared spectral-NN of the third type (weight sharing accross m for each h)
    def __init__(self,N,d,M,L,depth,width,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(spectralNNDeepshared3, self).__init__()
        self.N = N
        self.L = L
        self.depth = depth
        self.act_fn = act_fn
        self.init = init
        self.weight0 = torch.empty([2*L+1,width,d],dtype=torch.float32,requires_grad=True) #weights for the first hidden layer
        self.bias0 = torch.zeros([2*L+1,width,1],dtype=torch.float32,requires_grad=True) #biases for the first hidden layer
        self.weight = torch.empty([depth-1,2*L+1,width,width],dtype=torch.float32,requires_grad=True) #weights for the other hidden layers
        self.bias = torch.zeros([depth-1,2*L+1,width,1],dtype=torch.float32,requires_grad=True) #biases for the other hidden layer
        self.weight_final = torch.empty([M,2*L+1,width],dtype=torch.float32,requires_grad=True) #weights for the output layer
        self.bias_final = torch.zeros([M,2*L+1,1],dtype=torch.float32,requires_grad=True) #biases for the output layer
        self.xi = torch.empty([M,N+2*L],dtype=torch.float32,requires_grad=True) #the multipliers xi_{m,h}
        init(self.weight0)
        init(self.weight)
        init(self.weight_final)
        init(self.xi)
        self.params = list([self.weight0,self.bias0,self.weight,self.bias,self.weight_final,self.bias_final,self.xi])

    def first_step(self, u):
        u1 = self.act_fn(torch.einsum("jkl,ml -> jkm", self.weight0, u) + self.bias0)
        for i in range(self.depth-1):
            u1 = self.act_fn(torch.einsum("jkl, jlm -> jkm", self.weight[i], u1) + self.bias[i])
        return self.act_fn(torch.einsum("ijk,jkm -> ijm", self.weight_final, u1) + self.bias_final) #an object of size M x 2L+1 x D

    def iter_prod(self, i, G): ## iterated product with the coefficients in xi
        return torch.einsum("ij,ijk -> k", self.xi[:,i:(i+2*self.L+1)], G).reshape(1,-1)

    def forward(self, u):
        G = self.first_step(u)
        return torch.cat([self.iter_prod(i,G) for i in range(self.N)])