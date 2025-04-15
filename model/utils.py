import torch
from cmath import sqrt

def sigmoid_cons(param, dmin, dmax):
    """
    constrain the output physilogical parameters between *dmin* and *dmax*
    """
    return dmin+(torch.sigmoid(param))*(dmax-dmin)

def rice_exp(v, sigma):
    """
    Add the rician bias
    """
    t = v / sigma
    res= sigma*(sqrt(torch.pi/8)*
                    ((2+t**2)*torch.special.i0e(t**2/4)+
                    t**2*torch.special.i1e(t**2/4)))
    res = res.to(torch.float32)
    return res

def bio_exp(d1, d2, f, b):
    """ivim model"""
    v = f*torch.exp(-b*d1*1e-3+1e-6) + (1-f)*torch.exp(-b*d2*1e-3+1e-6)

    return v

def kurtosis(bval, D, K):
    """
    torch kurtosis function
    """

    X = torch.exp(-bval*D*1e-3+(bval*D*1e-3)**2*K/6+1e-6)

    return X
def gamma(bval, theta, K):
    """
    torch gamma function
    """
    X = torch.float_power(1+theta*bval*1e-3,-K)+1e-6
    return X
