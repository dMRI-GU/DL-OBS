from model import UNet2D
from tqdm import tqdm
import torch
from torch import nn
from utils import get_toy_datasets, rice_exp, get_lr, try_gpu, init_weights
import warnings
import numpy as np

def fit_one_epoch(net, b, batch_size, train_loader, val_loader, optimizer, device, epo):

    train_losses = 0 
    val_losses = 0
    net.train()

    with tqdm(total=batch_size, desc='Epoch :{}'.format(epo)) as pbar:
        for iteration, batch in enumerate(train_loader):
            batch_images = batch
            with torch.no_grad():
                batch_images =batch_images.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            out_maps = net(batch_images)
            M = rice_exp(out_maps, b)
            M = M.to(torch.float32)

            mseloss = nn.MSELoss()
            loss = mseloss(M, batch_images.view(M.shape).requires_grad_())

            loss.backward()
            optimizer.step()

            train_losses += loss.item()

            # pbar.set_postfix(**{'total_loss': train_losses / (iteration + 1), 
            #                     'lr'        : get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    for batch in val_loader:
        val_images = batch
        with torch.no_grad():
            val_images = val_images.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            out_maps = net(val_images)

            M = rice_exp(out_maps, b)
            M = M.to(torch.float32)

            mseloss = nn.MSELoss()
            val_loss = mseloss(M, val_images.view(M.shape).requires_grad_())

            val_losses += val_loss.item()
    print('The total validation loss is {}'.format(val_losses))
    print('Stop Validation')

    return val_losses

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    num_epochs = 80
    lr = 1e-5
    batch_size = 2
    in_channels = 3
    output_channels = 5
    b = torch.linspace(100, 2000, steps=in_channels)
    device = try_gpu()
    b.to(device)

    model = UNet2D(in_channels=in_channels, out_channels=output_channels)
    model.apply(init_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    data_set = get_toy_datasets(120, in_channels)  
    for i in range(num_epochs):
        train_set, val_set = torch.utils.data.random_split(data_set, lengths=[100, 20])  
        print(len(val_set))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
        val_losses = fit_one_epoch(model, b, batch_size, train_loader, val_loader, optimizer, device, i)
        scheduler.step(val_losses)


