from model.unet_parts import *
from model.utils import *

class UNet_MultiDecoders(nn.Module):
    def __init__(self, n_channels, b, rice=True,  bilinear=False, attention=False):
        super(UNet_MultiDecoders, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.b_values = b.reshape(1, len(b), 1, 1)
        self.rice = rice
        self.attention = attention

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        channels = [1024, 512, 256, 128, 64]

        if self.attention:
            self.decoder1 = Atten_Decoder(channels, factor)
            self.decoder2 = Atten_Decoder(channels, factor)
            self.decoder3 = Atten_Decoder(channels, factor)
            self.decoder4 = Atten_Decoder(channels, factor)
        else:    
            self.decoder1 = Decoder(channels, factor, self.bilinear)
            self.decoder2 = Decoder(channels, factor, self.bilinear)
            self.decoder3 = Decoder(channels, factor, self.bilinear)
            self.decoder4 = Decoder(channels, factor, self.bilinear)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        f_maps = [x4, x3, x2, x1]

        d_1 = self.decoder1(f_maps, x5)
        d_2 = self.decoder2(f_maps, x5)
        f = self.decoder3(f_maps, x5)
        sigma_g = self.decoder4(f_maps, x5)

        if torch.mean(d_1) < torch.mean(d_2):
            d_1, d_2 = d_2, d_1
            f = 1 - f

        d_1 = sigmoid_cons(d_1, 2, 2.4)
        d_2 = sigmoid_cons(d_2, 0.1, 0.5)
        f = sigmoid_cons(f, 0.5, 1.0)
        
        # (batch_size, 20, 200, 240)
        v = bio_exp(d_1, d_2, f, self.b_values)
        
        res = torch.zeros(v.shape)
        # add the rice-bias
        if self.rice:
            res = rice_exp(v, sigma_g)
            
            # only the correct the large b values
            #res[:, 0:10, :, :] = v[:, 0:10, :, :]
        else:
            res = v
    
        return res, d_1, d_2, f, sigma_g

