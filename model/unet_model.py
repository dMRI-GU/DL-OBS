""" Full assembly of the arts to form the complete network """
from model.diffusion_modeller import DiffusionModeller
from model.unet_parts import *
from model.utils import *
from cmath import sqrt
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_channels, input_sigma: bool, fitting_model:str,estimate_S0,estimate_sigma,use_true_sigma,feed_sigma, rice=True, bilinear=False,use_3D = False,learn_sigma_scaling = False):
        super(UNet, self).__init__()
        self.input_sigma = input_sigma
        self.n_channels = n_channels#20 or 60 if use_3D
        self.fitting_model = fitting_model
        self.use_3D = use_3D
        self.learn_sigma_scaling = learn_sigma_scaling
        self.estimate_S0 = estimate_S0#Boolean if AI will estimate b0-image
        self.estimate_sigma = estimate_sigma
        if fitting_model == 'biexp':
            self.n_classes = 3
        elif fitting_model == 'kurtosis':
            self.n_classes = 2
        elif fitting_model == 'gamma':
            self.n_classes = 2

        if use_3D:
            self.n_classes *= 3 #three times more parameters to predict20
        if not self.input_sigma or self.estimate_sigma:
            self.n_classes += 1#If no noise map was input to AI, then AI will predict one
            self.estimate_sigma = 'true'
        if self.estimate_S0:
            self.n_classes += 1

        self.bilinear = bilinear#For upsampling, default is False and instead ConvTranspose will be used
        self.rice = rice#If Rician bias is going to be added.
        self.feed_sigma = feed_sigma# If noise map is input to AI, then number of input channels is +1
        self.use_true_sigma = use_true_sigma
        if self.feed_sigma: add_channel = 1
        else: add_channel = 0

        if use_3D:#First layer
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
            self.sigma_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))#A learnable parameter
        else:
            self.sigma_scale = torch.tensor(1.0, requires_grad=False)# A constant

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

        DM = DiffusionModeller(self.input_sigma, self.fitting_model, self.estimate_S0, self.estimate_sigma,
                               self.use_true_sigma,
                               self.rice, self.use_3D, self.learn_sigma_scaling, self.n_classes)

        return DM.forward(logits, b, sigma_true, b0, scale_factor, self.sigma_scale)
