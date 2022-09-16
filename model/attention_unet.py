from model.unet_parts import *
from model.utils import *
from cmath import sqrt

class Atten_Unet(nn.Module):
    def __init__(self, n_channels, b, rice=True, bilinear=False):
        super(Atten_Unet, self).__init__()
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

        self.upconv1 = Up_conv(1024 // factor, 512) 
        self.atten1 = Attention_block(512, 512, 256)
        self.dbconv1 = DoubleConv(1024, 512)
        
        self.upconv2 = Up_conv(512 // factor, 256) 
        self.atten2 = Attention_block(256, 256, 128)
        self.dbconv2 = DoubleConv(512, 256)

        self.upconv3 = Up_conv(256 // factor, 128) 
        self.atten3 = Attention_block(128, 128, 64)
        self.dbconv3 = DoubleConv(256, 128)

        self.upconv4 = Up_conv(128 // factor, 64) 
        self.atten4 = Attention_block(64, 64, 32)
        self.dbconv4 = DoubleConv(128, 64)
   
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
       
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
        logits = torch.abs(self.outc(d2))

        d_1 = logits[:, 0:1, :, :]
        d_2 = logits[:, 1:2, :, :]
        f = logits[:, 2:3, :, :]
        sigma = logits[:, 3:4, :, :]

        # make sure D1 is the larger value between D1 and D2
        if torch.mean(d_1) < torch.mean(d_2):
            d_1, d_2 = d_2, d_1
            f = 1 - f 
        
        d_1 = sigmoid_cons(d_1, 2.0, 2.4)
        d_2 = sigmoid_cons(d_2, 0.1, 0.5)
        f = sigmoid_cons(f, 0.5, 1.0)

        # get the expectation of the clean images
        v = bio_exp(d_1, d_2, f, self.b_values)

        # add the rician bias
        if self.rice:
            res = rice_exp(v, sigma)
        else:
            res = v

        return res, d_1, d_2, f, sigma
    
    def pad_cat(self, s, b):
        """The feature map of s is smaller than that of b"""
        diffY = b.size()[2] - s.size()[2]
        diffX = b.size()[3] - s.size()[3]

        s = F.pad(s, [diffX // 2, diffX - diffX // 2,
                      diffY //2, diffY - diffY //2])

        ot = torch.cat([b, s], dim=1)
        return ot
