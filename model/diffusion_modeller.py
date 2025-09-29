import torch
import numpy as np
from model.utils import *
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModeller:
    def __init__(self, input_sigma: bool, fitting_model:str,estimate_S0,estimate_sigma,use_true_sigma, rice,use_3D,learn_sigma_scaling, n_classes):
        super().__init__()
        self.input_sigma = input_sigma
        self.fitting_model = fitting_model
        self.use_3D = use_3D#-#
        self.n_classes = n_classes
        self.estimate_sigma = estimate_sigma
        self.use_true_sigma = use_true_sigma
        self.rice = rice
        self.estimate_S0 = estimate_S0#Boolean if AI will estimate b0-image
        self.learn_sigma_scaling = learn_sigma_scaling


        self.clippedsofplus = ScaledClippedSoftplus()
    def forward(self, logits,b,sigma_true,b0,scale_factor, sigma_scale):

        num_diffusion = 3 if self.use_3D else 1
        num_par = self.n_classes

        # n_classes may not count noise map and b0-image
        if not self.estimate_sigma:
            num_par += 1
        if not self.estimate_S0:
            num_par += 1

        par_collect = torch.zeros(size=(
        num_par, logits.shape[0], *logits.shape[-2:]))  # collect all parameters in one array (num_par,num_batches,H,W)
        par_name_list = [None] * num_par  # [None,None,None...]

        imag_collect = torch.zeros(size=(num_diffusion, logits.shape[0], np.max(b.shape), *logits.shape[-2:]),
                                   device=logits.device)  # (num_diffusion_directions, num_batches, num_diffusion_levels, H,W)

        estimated_sigma = None
        if self.estimate_sigma:
            estimated_sigma = self.clippedsofplus(logits[:, slice(-1, None), :, :])
        if self.use_true_sigma or (self.input_sigma and not self.estimate_sigma):
            sigma_true[sigma_true == 0.] = 1e-8  # To avoid overflow
            if self.learn_sigma_scaling:
                sigma_scale = sigma_scale.to(device=logits.device)
                sigma_scale = F.relu(sigma_scale)
                sigma_final = sigma_true * sigma_scale
            else:
                sigma_final = sigma_true
        elif self.estimate_sigma and not self.use_true_sigma:
            sigma_final = estimated_sigma
        else:
            assert False, 'Error occured in res_atten_unet no sigma?'
        sigma_final[sigma_final == 0.] = 1e-8

        par_collect[-1] = sigma_final[:, 0]
        par_name_list[-1] = 'final_sigma'

        if self.estimate_S0:
            if not self.input_sigma or self.estimate_sigma:
                s0 = sigmoid_cons(logits[:, slice(-2, -1), :, :], 0.001,
                                  1.4)  # last index is sigma, second last index is s0
            else:
                s0 = sigmoid_cons(logits[:, slice(-1, None), :, :], 0.001,
                                  1.4)  # last index is s0, there is no predicted noise map
        else:
            s0 = b0
        par_collect[-2] = s0[:, 0]
        par_name_list[-2] = 's0'

        for index in range(num_diffusion):
            if self.fitting_model == 'biexp':

                d_1 = logits[:, 3 * index + 0:3 * index + 1, :, :]  # shape batch,1,200,240
                d_2 = logits[:, 3 * index + 1:3 * index + 2, :, :]
                f = logits[:, 3 * index + 2:3 * index + 3, :, :]


                d_1 = sigmoid_cons(d_1, 0, 4)  # testa utan också
                d_2 = sigmoid_cons(d_2, 0, 1)
                f = sigmoid_cons(f, 0.1, 0.9)

                # collect

                par_collect[3 * index + 0] = d_1[:, 0]
                par_collect[3 * index + 1] = d_2[:, 0]
                par_collect[3 * index + 2] = f[:, 0]
                par_name_list[3 * index + 0] = 'D1'
                par_name_list[3 * index + 1] = 'D2'
                par_name_list[3 * index + 2] = 'f'

                # get the expectation of the clean images

                v = bio_exp(d_1, d_2, f, b)

                if self.estimate_S0:
                    v = (s0 * v)
                else:
                    v = (s0 * v) / (scale_factor.view(-1, 1, 1, 1))

                if self.rice:
                    res = F.relu(rice_exp(v, sigma_final))  # +  1 / scale_factor.view(-1, 1, 1, 1)
                else:
                    res = F.relu(v)  # +  1 / scale_factor.view(-1, 1, 1, 1)
                imag_collect[index] = res
            elif self.fitting_model == 'kurtosis':

                d = logits[:, 2 * index + 0:2 * index + 1, :, :]
                k = logits[:, 2 * index + 1:2 * index + 2, :, :]

                d = sigmoid_cons(d, 0, 4)
                k = sigmoid_cons(k, 0, 5)

                par_collect[2 * index + 0] = d[:, 0]
                par_collect[2 * index + 1] = k[:, 0]
                par_name_list[2 * index + 0] = 'D'
                par_name_list[2 * index + 1] = 'K'

                # get the expectation of the clean images
                v = kurtosis(b, D=d, K=k)
                v = v.to(torch.float64)
                if self.estimate_S0:
                    v = (s0 * v)
                else:
                    v = (s0 * v) / (scale_factor.view(-1, 1, 1, 1))

                if self.rice:
                    res = rice_exp(v, sigma_final)
                else:
                    res = v
                imag_collect[index] = res
            elif self.fitting_model == 'gamma':
                theta = logits[:, 2 * index + 0:2 * index + 1, :, :]
                k = logits[:, 2 * index + 1:2 * index + 2, :, :]

                theta = sigmoid_cons(theta, 0, 10)
                k = sigmoid_cons(k, 0, 20)

                par_collect[2 * index + 0] = k[:, 0]
                par_collect[2 * index + 1] = theta[:, 0]
                par_name_list[2 * index + 0] = 'K'
                par_name_list[2 * index + 1] = 'Theta'

                # get the expectation of the clean images
                v = gamma(bval=b, theta=theta, K=k)

                if self.estimate_S0:
                    v = (s0 * v)
                else:
                    v = (s0 * v) / (scale_factor.view(-1, 1, 1, 1))
                if self.rice:
                    res = rice_exp(v, sigma_final)
                else:
                    res = v
                imag_collect[index] = res
        imag_collect_cat = torch.cat([imag_collect[i] for i in range(num_diffusion)], dim=1)  # Concatenate along dim=1
        return (imag_collect_cat, {'parameters': par_collect, 'sigma': sigma_final * scale_factor.view(-1, 1, 1, 1),
                                  'names': par_name_list, **({'estimated_sigma': estimated_sigma * scale_factor.view(-1, 1,
                                                                                                                     1,
                                                                                                                     1)} if estimated_sigma is not None else {})})





class ScaledClippedSoftplus(nn.Module):
    def __init__(self, max_value=0.5, beta=1.0, threshold=20.0):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)
        self.max_value = max_value

    def forward(self, x):
        out = self.softplus(x)+0.0001
        return torch.clamp(out, max=self.max_value)