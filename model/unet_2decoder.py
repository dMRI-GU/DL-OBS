
from model.unet_parts import *


class UNet_2Decoders(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_2Decoders, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1_1 = Up(1024, 512 // factor, bilinear)
        self.up2_1 = Up(512, 256 // factor, bilinear)
        self.up3_1 = Up(256, 128 // factor, bilinear)
        self.up4_1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.up1_2 = Up(1024, 512 // factor, bilinear)
        self.up2_2 = Up(512, 256 // factor, bilinear)
        self.up3_2 = Up(256, 128 // factor, bilinear)
        self.up4_2 = Up(128, 64, bilinear)
        self.outg = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1_1(x5, x4)
        x = self.up2_1(x, x3)
        x = self.up3_1(x, x2)
        x = self.up4_1(x, x1)
        logits = self.outc(x)

        s = self.up1_2(x5, x4)
        s = self.up2_2(s, x3)
        s = self.up3_2(s, x2)
        s = self.up4_2(s, x1)        

        sigma_g = self.outg(s)

        return logits, sigma_g