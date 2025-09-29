import matplotlib
import torch
from cmath import sqrt
import numpy as np
from IPython import embed
from math import sqrt, pi
from scipy.special import i0e, i1e

def sigmoid_cons(param, dmin, dmax):
    """
    constrain the output physilogical parameters between *dmin* and *dmax*
    """
    return dmin+(torch.sigmoid(param))*(dmax-dmin)

def arctan_cons(param,dmin,dmax):

    return  (dmax - dmin) / torch.pi * torch.atan(param) + (dmin + dmax) / 2

def rice_exp(v, sigma):
    """
    Add the rician bias
    """
    t = v / sigma

    if isinstance(v, np.ndarray):
        res = sigma * (sqrt(pi / 8) *
                       ((2 + t ** 2) * i0e(t ** 2 / 4) +
                        t ** 2 * i1e(t ** 2 / 4)))

    elif isinstance(v, torch.Tensor):
        res= sigma*(sqrt(torch.pi/8)*
                        ((2+t**2)*torch.special.i0e(t**2/4)+
                        t**2*torch.special.i1e(t**2/4)))
        res = res.to(t.dtype)
    return res

def bio_exp(d1, d2, f, b):
    if isinstance(b, np.ndarray):
        v = f*np.exp(-b*d1*1e-3+1e-6) + (1-f)*np.exp(-b*d2*1e-3+1e-6)


    elif isinstance(b, torch.Tensor):

        v = f*torch.exp(-b*d1*1e-3+1e-6) + (1-f)*torch.exp(-b*d2*1e-3+1e-6)

    return v

def kurtosis(bval, D, K):
    """
    torch kurtosis function
    """
    if isinstance(bval, np.ndarray):
        # Use numpy
        X = np.exp(-bval * D * 1e-3 + (bval * D * 1e-3) ** 2 * K / 6 + 1e-6)
    elif isinstance(bval, torch.Tensor):
        # Use torch
        X = torch.exp(-bval * D * 1e-3 + (bval * D * 1e-3) ** 2 * K / 6 + 1e-6)

    return X
def gamma(bval, theta, K):
    """
    torch gamma function
    """
    log_gamma = -K * torch.log(torch.clamp(1 + theta * bval * 1e-3, min=1e-3))  # + log(1e-6) if needed
    X = torch.exp(log_gamma)
    return X
