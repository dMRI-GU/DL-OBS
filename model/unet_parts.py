""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvResidual(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),

        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.residual_final_relu =  nn.ReLU(inplace=True)
    def forward(self, x):

        x1 =  self.double_conv(x)
        x2 = self.residual_conv(x)

        return self.residual_final_relu(x1+x2)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Res_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvResidual(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # upsampled feature map might be smaller than the original one
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.up(x)

class Decoder(nn.Module):
    """Stack several upsampling layers"""
    def __init__(self, channels, factor, bilinear):
        """
        channels - e.g [1024, 512, 256, 128, 64]
        """
        super().__init__()
        self.up1 = Up(channels[0], channels[1] // factor, bilinear)
        self.up2 = Up(channels[1], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[3] // factor, bilinear)
        self.up4 = Up(channels[3], channels[4], bilinear)
        self.outc = OutConv(channels[4], 1)

    def forward(self, f_maps, x5):
        """
        f_maps: [x4, x3, x2, x1]
        """
        x = self.up1(x5, f_maps[0])
        x = self.up2(x, f_maps[1])
        x = self.up3(x, f_maps[2])
        x = self.up4(x, f_maps[3])

        return torch.abs(self.outc(x))

class Atten_Decoder(nn.Module):
    def __init__(self, channels, factor):
        """channels - e.g [1024, 512, 256, 128, 64]"""
        super(Atten_Decoder, self).__init__()

        self.upconv1 = Up_conv(channels[0] // factor, channels[1]) 
        self.atten1 = Attention_block(channels[1], channels[1], channels[2])
        self.dbconv1 = DoubleConv(channels[0], channels[1])
        
        self.upconv2 = Up_conv(channels[1] // factor, channels[2]) 
        self.atten2 = Attention_block(channels[2], channels[2], channels[3])
        self.dbconv2 = DoubleConv(channels[1], channels[2])

        self.upconv3 = Up_conv(channels[2] // factor, channels[3]) 
        self.atten3 = Attention_block(channels[3], channels[3], channels[4])
        self.dbconv3 = DoubleConv(channels[2], channels[3])

        self.upconv4 = Up_conv(channels[3] // factor, channels[4]) 
        self.atten4 = Attention_block(channels[4], channels[4], channels[4] // 2)
        self.dbconv4 = DoubleConv(channels[3], channels[4])

        self.outc = OutConv(channels[4], 1)

    def forward(self, f_maps, x5):       
        x4, x3, x2, x1 = f_maps

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
        return torch.abs(self.outc(d2))

    def pad_cat(self, s, b):
        """The feature map of s is smaller than that of b"""
        diffY = b.size()[2] - s.size()[2]
        diffX = b.size()[3] - s.size()[3]

        s = F.pad(s, [diffX // 2, diffX - diffX // 2,
                      diffY //2, diffY - diffY //2])

        ot = torch.cat([b, s], dim=1)
        return ot

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self,  F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        diffY = x1.size()[2] - g1.size()[2]
        diffX = x1.size()[3] - g1.size()[3]

        g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)

        return alpha * x
