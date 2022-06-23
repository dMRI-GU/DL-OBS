from cmath import sqrt
import torch
import os

def rice_exp(output_map, b):
    v = biexp(output_map, b)
    sigma_g = output_map[:, 4, :, :, :]

    t= v /sigma_g
    res= sigma_g*(sqrt(torch.pi/8)*
                    ((2+t**2)*torch.special.i0e(t**2/4)+
                     t**2*torch.special.i1e(t**2/4)))

    return res

def biexp(output_map, b):
    '''
    Reconstuct the denoised signal using the output parameter map
    '''
    s_0, d_1, d_2, f = output_map[:, 0, :, :, :], output_map[:, 1, :, :, :], output_map[:, 2, :, :, :], output_map[:, 3, :, :, :]

    num_slices, _,  h, w = s_0.shape

    'vb (num of slices, b values, h, w)'
    vb = torch.ones((num_slices, len(b), h, w), device=try_gpu())

    for i in range(len(b)):
        vb[:, i, :, :] = b[i]

    return s_0 *(f * torch.exp(- vb * d_1  * 1e-3) + (1 - f) * torch.exp(- vb * d_2 * 1e-3))

def sigmoid_cons(params,cons):
    """
    params: parameter array
    cons: constraints cons[0]: lower bound cons[1]: upper bound
    """
    return cons[0]+torch.sigmoid(params)*(cons[1]-cons[0])

def init_weights(m):
    'Xaiver initialization'
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

def get_toy_datasets(num_images, in_channels):
    '''
    simulated data sets for testing
    '''
    return torch.randn(num_images, in_channels, 1, 128, 128)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def try_gpu(i=0):
    '''
    If GPU exists, return the ith GPU, else return cpu
    '''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def number_of_features_per_level(init_channel_number, num_levels):
    '''
    Get the number of kernels per level.
    e.g. num_levels = 4, init_channel_number = 64.
        return [64, 128, 256, 512]
    '''
    return [init_channel_number * 2 ** k for k in range(num_levels)]