import torch as to
from math import sqrt, pi

def strain(lam):
    x1 = .5*(lam[:,0]**2+lam[:,1]**2)
    x3 = lam[:,0]*(-lam[:,0]**2 + 3*lam[:,1]**2)/12*sqrt(3.)
    VE = x1**3 - 6*x3**2
    #VE = to.where(VE>1e-12, to.log(VE), to.tensor(0.))
    VE = .5*to.log(6*pi**4*VE)
    return VE.float()


def unitary(x):
    return (0.*x[:,0]).float()
