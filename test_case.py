import torch
from utils import rice_exp, try_gpu
from model import UNet2D

if __name__ == '__main__':
    device = try_gpu()
    print(device)
    batch_size = 3
    in_channels = 3
    output_channels = 5
    num_epochs = 10
    h, w = 128, 128
    x = torch.randn(batch_size, in_channels, 1, h, w)
    x.to(device)

    print('x size: {}'.format(x.size()))

    model = UNet2D(in_channels=in_channels, out_channels=output_channels, layer_order='bcr')

    out = model(x)

    print('output size: {}'.format(out.size()))

    b = torch.linspace(100, 2000, steps=in_channels)
    print(b.size())

    M = rice_exp(out, b)
    print('Expectation value size: {}'.format(M.size()))

    x = x.view(M.shape)
    M = M.type(torch.int64)
    koss = torch.nn.MSELoss()

    print(koss(M, x))




    