""" Full assembly of the arts to form the complete network """
from model.unet_parts import *
from model.utils import *
from cmath import sqrt


class UNet(nn.Module):
    def __init__(self, n_channels, input_sigma: bool, fitting_model:str, rice=True, bilinear=False):
        super(UNet, self).__init__()
        self.input_sigma = input_sigma
        self.n_channels = n_channels
        self.fitting_model = fitting_model
        ####################################################################################
        ####################################################################################
        ####################################################################################
        ####################################################################################
        if fitting_model == 'biexp':
            self.n_classes = 3
        elif fitting_model == 'kurtosis':
            self.n_classes = 2
        elif fitting_model == 'gamma':
            self.n_classes = 2
        ####################################################################################
        if not self.input_sigma:
            self.n_classes += 1

        self.bilinear = bilinear
        self.rice = rice

        self.inc = DoubleConv(n_channels, 64)
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
        self.sigma_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, x,b,b0,sigma_true, scale_factor):
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

        if self.fitting_model == 'biexp':
            d_1 = logits[:, 0:1, :, :]
            d_2 = logits[:, 1:2, :, :]
            f = logits[:, 2:3, :, :]
            # sigma = logits[:, 3:4, :, :]
            if self.input_sigma:
                sigma_true[sigma_true == 0.] = 1e-8
                sigma_scale = self.sigma_scale.to(device=d_1.device)
                sigma_scale = F.relu(sigma_scale)
                sigma_final = sigma_true * sigma_scale
            else:
                sigma_final = sigmoid_cons(logits[:, 3:4, :, :],0.001,1)

            sigma_final[sigma_final == 0.] = 1e-8
            # make sure D1 is the larger value between D1 and D2
            if torch.mean(d_1) < torch.mean(d_2):
                d_1, d_2 = d_2, d_1
                f = 1 - f

            d_1 = sigmoid_cons(d_1, 0, 4)  # testa utan också
            d_2 = sigmoid_cons(d_2, 0, 1)
            f = sigmoid_cons(f, 0.1, 0.9)
            # get the expectation of the clean images

            v = bio_exp(d_1, d_2, f, b)

            v = (b0 * v) / (scale_factor.view(-1, 1, 1, 1))
            if self.rice:
                res = rice_exp(v, sigma_final)
            else:
                res = v
            return res, {'d1': d_1, 'd2': d_2, 'f': f, 'sigma': sigma_final * scale_factor.view(-1, 1, 1, 1)}

        elif self.fitting_model == 'kurtosis':

            d = logits[:, 0:1, :, :]
            k = logits[:, 1:2, :, :]
            if self.input_sigma:
                sigma_true[sigma_true == 0.] = 1e-8
                sigma_scale = self.sigma_scale.to(device=d.device)
                sigma_scale = F.relu(sigma_scale)
                sigma_final = sigma_true * sigma_scale
            else:
                sigma_final = sigmoid_cons(logits[:, 2:3, :, :],0.001,1)

            sigma_final[sigma_final == 0.] = 1e-8
            # make sure D1 is the larger value between D1 and D2

            d = sigmoid_cons(d, 0, 4)
            k = sigmoid_cons(k, 0, 1)
            # get the expectation of the clean images
            v = kurtosis(b, D=d, K=k)

            v = (b0 * v) / (scale_factor.view(-1, 1, 1, 1))
            if self.rice:
                res = rice_exp(v, sigma_final)
            else:
                res = v
            return res, {'D': d, 'K': k, 'sigma': sigma_final * scale_factor.view(-1, 1, 1, 1)}
        elif self.fitting_model == 'gamma':
            theta = logits[:, 0:1, :, :]
            k = logits[:, 1:2, :, :]
            if self.input_sigma:
                sigma_true[sigma_true == 0.] = 1e-8
                sigma_scale = self.sigma_scale.to(device=k.device)
                sigma_scale = F.relu(sigma_scale)
                sigma_final = sigma_true * sigma_scale
            else:
                sigma_final = sigmoid_cons(logits[:, 2:3, :, :],0.001,1)

            sigma_final[sigma_final == 0.] = 1e-8
            # make sure D1 is the larger value between D1 and D2

            theta = sigmoid_cons(theta, 0, 10)
            k = sigmoid_cons(k, 0, 20)
            # get the expectation of the clean images
            v = gamma(bval=b, theta=theta, K=k)

            v = (b0 * v) / (scale_factor.view(-1, 1, 1, 1))
            if self.rice:
                res = rice_exp(v, sigma_final)
            else:
                res = v
            return res, {'Theta': theta, 'K': k, 'sigma': sigma_final * scale_factor.view(-1, 1, 1, 1)}






