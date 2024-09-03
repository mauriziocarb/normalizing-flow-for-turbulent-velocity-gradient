import torch as to
from math import pi

def strain(x):
    #first invariant
    return (-5*x[0]-.1*x[1]**2).float()

def vorticity(x):
    return (3*x[:,1]).float()

def gradient(x):
    return (-5*x[0] + 3*x[1]).float()

def normal(x):
    #rescaled S eigenvalues as input
    #return (-2.5*to.einsum("si,si->s", x, x)-.5*to.log(to.tensor((2*pi)**2/25, device=x.device))).float()
    rho = -.5*(x**2) ###- to.log(y*(1. - y))
    return (rho.sum(axis=-1))#.float()

def log_normal(x):
    #rescaled S eigenvalues as input
    y = .5*to.log(x[:,0]**2)
    return (-(y+2)**2/10-y).float()



def normal_01(y):
    #input 0-1, outp Gaussian
    y = y.double()
    x = to.log(y/(1. - y))
    rho = -.5*(x**2) - to.log(y*(1. - y))
    return (rho.sum(axis=-1)).float()
