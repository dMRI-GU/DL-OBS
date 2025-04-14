""" Full assembly of the arts to form the complete network """
from model.unet_parts import *
from model.utils import *
from cmath import sqrt
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_channels, input_sigma: bool, fitting_model:str,estimate_S0,feed_sigma, rice=True, bilinear=False,use_3D = False,learn_sigma_scaling = False):
        super(UNet, self).__init__()
        self.input_sigma = input_sigma
        self.n_channels = n_channels
        self.fitting_model = fitting_model
        self.use_3D = use_3D
        self.learn_sigma_scaling = learn_sigma_scaling
        self.estimate_S0 = estimate_S0
        if fitting_model == 'biexp':
            self.n_classes = 3
        elif fitting_model == 'kurtosis':
            self.n_classes = 2
        elif fitting_model == 'gamma':
            self.n_classes = 2

        if use_3D:
            self.n_classes *= 3
        ####################################################################################
        if not self.input_sigma:
            self.n_classes += 1
        if self.estimate_S0:
            self.n_classes += 1

        self.bilinear = bilinear
        self.rice = rice
        self.feed_sigma = feed_sigma
        if self.feed_sigma: add_channel = 1
        else: add_channel = 0

        if use_3D:
            self.inc = nn.Sequential(
                DoubleConv(n_channels + add_channel, 64 * 3),
                DoubleConv(64 * 3, 64 * 2),
                DoubleConv(64 * 2, 64)
            )
        else:
            self.inc = DoubleConv(n_channels + add_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)
        if self.learn_sigma_scaling:
            self.sigma_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        else:
            self.sigma_scale = torch.tensor(1.0, requires_grad=False)

    def forward(self, x,b,b0,sigma_true, scale_factor):


        if self.feed_sigma:  x = torch.cat([x, sigma_true], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if torch.isnan(logits).sum() > 0 or torch.max(logits) > 1e10:
            print(f'-Warning: Logits contained {torch.isnan(logits).sum().item()} NaN values and {torch.max(logits)} as maximum value.\n')

        num_diffusion = 3 if self.use_3D else 1
        num_par = self.n_classes
        if self.input_sigma:
            num_par+=1
        if not self.estimate_S0:
            num_par+=1
        par_collect = torch.zeros(size=(num_par, logits.shape[0], *logits.shape[-2:]))
        par_name_list = [None] * num_par

        imag_collect = torch.zeros(size=(num_diffusion, logits.shape[0], np.max(b.shape), *logits.shape[-2:]),
                                   device=logits.device)
        if self.input_sigma:
            sigma_true[sigma_true == 0.] = 1e-8
            if self.learn_sigma_scaling:
                sigma_scale = self.sigma_scale.to(device=logits.device)
                sigma_scale = F.relu(sigma_scale)
                sigma_final = sigma_true * sigma_scale
            else:
                sigma_final = sigma_true
        else:
            sigma_final = sigmoid_cons(logits[:, slice(-1, None), :, :], 0.01, 1)
        sigma_final[sigma_final == 0.] = 1e-8

        par_collect[-1] = sigma_final[:, 0]
        par_name_list[-1] = 'sigma'

        if self.estimate_S0:
            if not self.input_sigma: s0 = sigmoid_cons(logits[:, slice(-2, -1), :, :],0.001,1.4)
            else: s0 = sigmoid_cons(logits[:, slice(-1, None), :, :],0.001,1.4)
        else:
            s0 = b0
        par_collect[-2] = s0[:, 0]
        par_name_list[-2] = 's0'
        for index in range(num_diffusion):
            if self.fitting_model == 'biexp':

                d_1 = logits[:, 3*index + 0:3*index + 1, :, :]#shape batch,1,200,240
                d_2 = logits[:, 3*index + 1:3*index + 2, :, :]
                f =   logits[:, 3*index + 2:3*index + 3, :, :]

                # make sure D1 is the larger value between D1 and D2
                #if torch.mean(d_1) < torch.mean(d_2):
                #    d_1, d_2 = d_2, d_1
                #    f = 1 - f

                d_1 = sigmoid_cons(d_1, 0, 4)  # testa utan också
                d_2 = sigmoid_cons(d_2, 0, 1)
                f = sigmoid_cons(f, 0.1, 0.9)

                #collect

                par_collect[3 * index + 0] = d_1[:,0]
                par_collect[3 * index + 1] = d_2[:,0]
                par_collect[3 * index + 2] = f[:,0]
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
                    res =  F.relu(rice_exp(v, sigma_final)) #+  1 / scale_factor.view(-1, 1, 1, 1)

                else:
                    res = F.relu(v) #+  1 / scale_factor.view(-1, 1, 1, 1)

                imag_collect[index] = res
            elif self.fitting_model == 'kurtosis':

                d = logits[:, 2*index + 0:2*index + 1, :, :]
                k = logits[:, 2*index + 1:2*index + 2, :, :]


                d = sigmoid_cons(d, 0, 4)
                k = sigmoid_cons(k, 0, 1)

                par_collect[2 * index + 0] = d[:,0]
                par_collect[2 * index + 1] = k[:,0]
                par_name_list[2 * index + 0] = 'D'
                par_name_list[2 * index + 1] = 'K'


                # get the expectation of the clean images
                v = kurtosis(b, D = d,K = k)
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
                theta = logits[:, 2*index + 0:2*index + 1, :, :]
                k =     logits[:, 2*index + 1:2*index + 2, :, :]

                theta = sigmoid_cons(theta, 0, 10)
                k = sigmoid_cons(k, 0, 20)



                par_collect[2 * index + 0] = k[:,0]
                par_collect[2 * index + 1] = theta[:,0]
                par_name_list[2 * index + 0] = 'K'
                par_name_list[2 * index + 1] = 'Theta'

                # get the expectation of the clean images
                v = gamma(bval=b, theta=theta,K=k)

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
        return imag_collect_cat, {'parameters': par_collect, 'sigma': sigma_final * scale_factor.view(-1, 1, 1, 1), 'names':par_name_list}







