""" Full assembly of the parts to form the complete network """
from model.unet_parts import *
from model.utils import *
from cmath import sqrt


class UNet(nn.Module):
    def __init__(self, n_channels, b, rice = True, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 4
        self.b_values = b.reshape(1, len(b), 1, 1)
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
        self.outc = OutConv(64, 4)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = torch.abs(self.outc(x))

        d_1 = logits[:, 0:1, :, :]
        d_2 = logits[:, 1:2, :, :]
        f = logits[:, 2:3, :, :]
        sigma = logits[:, 3:4, :, :]
        sigma = torch.abs(sigma)

        # make sure D1 is the larger value between D1 and D2
        if torch.mean(d_1) < torch.mean(d_2):
            d_1, d_2 = d_2, d_1
            f = 1 - f 

        d_1 = self.sigmoid_cons(d_1, 2., 2.4)
        d_2 = self.sigmoid_cons(d_2, 0.1, 0.5)
        f = self.sigmoid_cons(f, 0.5, 1.0)

        #v = bio_exp(d_1, d_2, f, self.b_values)
        v = bio_exp(d_1, d_2, f, self.b_values)

        # add the rice-bias
        if self.rice:
            res = rice_exp(v, sigma)
        else:
            res = v

        return res, d_1, d_2, f, sigma

    def sigmoid_cons(self, param, dmin, dmax):
        """
        constrain the output physilogical parameters into a certain domain
        """
        return dmin+torch.sigmoid(param)*(dmax-dmin)

