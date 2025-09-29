from model.unet_parts import *
from model.utils import *
from cmath import sqrt
import numpy as np
from model.diffusion_modeller import DiffusionModeller

class Res_Atten_Unet(nn.Module):
    def __init__(self, n_channels, input_sigma: bool, fitting_model:str,estimate_S0,estimate_sigma,use_true_sigma,feed_sigma, rice=True, bilinear=False,use_3D = False,learn_sigma_scaling = False):
        super(Res_Atten_Unet, self).__init__()
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
            self.n_classes *= 3#three times more parameters to predict20
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

        if use_3D:
            self.inc = nn.Sequential(
                DoubleConv(n_channels+add_channel, 64 * 3),
                DoubleConv(64 * 3, 64 * 2),
                DoubleConv(64 * 2, 64)
            )
        else:
            self.inc = DoubleConv(n_channels+add_channel, 64)
        self.down1 = Res_Down(64, 128)
        self.down2 = Res_Down(128, 256)
        self.down3 = Res_Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Res_Down(512, 1024 // factor)

        self.upconv1 = Up_conv(1024 // factor, 512)
        self.atten1 = Attention_block(512, 512, 256)
        self.dbconv1 = DoubleConvResidual(1024, 512)

        self.upconv2 = Up_conv(512 // factor, 256)
        self.atten2 = Attention_block(256, 256, 128)
        self.dbconv2 = DoubleConvResidual(512, 256)

        self.upconv3 = Up_conv(256 // factor, 128)
        self.atten3 = Attention_block(128, 128, 64)
        self.dbconv3 = DoubleConvResidual(256, 128)

        self.upconv4 = Up_conv(128 // factor, 64)
        self.atten4 = Attention_block(64, 64, 32)
        self.dbconv4 = DoubleConvResidual(128, 64)

        self.outc = OutConv(64, self.n_classes)
        if self.learn_sigma_scaling:
            self.sigma_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # A learnable parameter
        else:
            self.sigma_scale = torch.tensor(1.0, requires_grad=False)  # A constant

    def forward(self, x,b,b0,sigma_true, scale_factor):


        if self.feed_sigma:  x = torch.cat([x, sigma_true], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d5 = self.upconv1(x5)
        x4 = self.atten1(d5, x4)
        d5 = self.pad_cat(d5, x4)
        d5 = self.dbconv1(d5)

        d4 = self.upconv2(d5)
        x3 = self.atten2(d4, x3)
        d4 = self.pad_cat(d4, x3)
        d4 = self.dbconv2(d4)

        d3 = self.upconv3(d4)
        x2 = self.atten3(d3, x2)
        d3 = self.pad_cat(d3, x2)
        d3 = self.dbconv3(d3)

        d2 = self.upconv4(d3)
        x1 = self.atten4(d2, x1)
        d2 = self.pad_cat(d2, x1)
        d2 = self.dbconv4(d2)
        logits = self.outc(d2)
        if torch.isnan(logits).sum() > 0 or torch.max(logits) > 1e10:
            print(
                f'-Warning: Logits contained {torch.isnan(logits).sum().item()} NaN values and {torch.max(logits)} as maximum value.\n')

        DM = DiffusionModeller(self.input_sigma, self.fitting_model,self.estimate_S0,self.estimate_sigma,self.use_true_sigma,
                               self.rice,self.use_3D,self.learn_sigma_scaling, self.n_classes)


        return DM.forward(logits,b,sigma_true,b0,scale_factor,self.sigma_scale)


    def pad_cat(self, s, b):
        """The feature map of s is smaller than that of b"""
        diffY = b.size()[2] - s.size()[2]
        diffX = b.size()[3] - s.size()[3]

        s = F.pad(s, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        ot = torch.cat([b, s], dim=1)
        return ot
