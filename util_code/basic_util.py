import numpy as np
import torch as to
import torch.distributed as dist
from torch.utils.data import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
import pickle
import sys

##################################################
### BASIC MATRIX UTILS
##################################################

def shufflerow(tensor, axis):
    row_perm = to.rand(tensor.shape[:axis+1]).argsort(axis)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def to_matrix(x):
    A = to.zeros([x.shape[0],9], device=x.device)
    A[:,:8] = x
    A[:,8] = -x[:,0]-x[:,4]
    A = A.reshape(-1,3,3)
    A -= to.einsum("skk,ij->sij", A, to.eye(3,device=x.device))/3.
#####    A = .5*(A + to.transpose(A,-1,-2))
    return A.float()


def to_matrix_extended(x):
    A = to.zeros([x.shape[0],x.shape[1],9], dtype=x.dtype, device=x.device)
    A[...,:8] = x
    A[..., 8] =-x[...,0]-x[...,4]
    A = A.reshape(A.shape[0],A.shape[1],3,3)
    A-= to.einsum("tskk,ij->tsij", A, to.eye(3,device=x.device))/3.
    return A

##################################################
### DERIVATIVES
##################################################

def gradient(y, x):
    
    s = y.shape
    assert len(s) == 1
    dydx = to.autograd.grad(y, x, 
    grad_outputs=y.data.new(y.shape).fill_(1),
    create_graph=True, retain_graph=True, allow_unused=True)[0]

    ###dydx = to.transpose(dydx, 0,1) #samples, N grad
    return dydx

def divergence(y, x):
    
    s = y.shape
    assert s == x.shape

    N_samples = s[0]
    N = s[-1]

####try optimization here x[:,i]
    div = to.zeros(N_samples, device=y.device)
    for i in range(N):
        z = y[:,i]
        div += to.autograd.grad(z, x, 
        grad_outputs=z.data.new(z.shape).fill_(1),
        create_graph=True, retain_graph=True, allow_unused=True)[0][:,i]

    return div




