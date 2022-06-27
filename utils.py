from cmath import sqrt
from torch.utils.data import Dataset
import torch

class post_processing():
    '''
    This class include the post processing function to evaluate the trained model
    '''
    def __init__(self, net, device):
        '''
        net - trainde neural network
        device - used the device for training
        '''
        super().__init__()
        self.net = net
        self.device = device
    
    def evaluate(self, val_loader, b):
        '''
        Get the validation loss during training
        '''
        loss = torch.nn.MSELoss()
        self.net.eval()
        val_losses = 0

        for batch in val_loader:
            images = batch['image']
            images = images.to(self.device)

            out = self.net(images)
            s_0, d_1, d_2, f, sigma_g = self.parameter_maps(out_maps=out)
            M = self.rice_exp(s_0, d_1, d_2, f, sigma_g, b)
            val_losses += loss(M, images).item()
        
        return val_losses


    def rice_exp(self, s_0, d_1, d_2, f, sigma_g, b):
        'Get the expectation of the signal using denoised signal and std.'
        v = self.biexp(s_0, d_1, d_2, f, b)

        t = v /sigma_g
        res= sigma_g*(sqrt(torch.pi/8)*
                        ((2+t**2)*torch.special.i0e(t**2/4)+
                        t**2*torch.special.i1e(t**2/4)))

        return res.to(torch.float32)
    
    def parameter_maps(self, out_maps):
        'Get the parameter maps from the output'
        s_0, d_1 = out_maps[:, 0:1, :, :], out_maps[:, 1:2, :, :]
        d_2, f, sigma_g = out_maps[:, 2:3, :, :], out_maps[:, 3:4, :, :], out_maps[:, 4:5, :, :]

        return s_0, d_1, d_2, f, sigma_g

    def biexp(self, s_0, d_1, d_2, f, b):
        '''
        Reconstuct the denoised signal using the output parameter maps
        '''
        num_slices, _, h, w = s_0.shape

        'vb (num of slices, b values, h, w)'
        vb = torch.zeros((num_slices, len(b), h, w))
        vb = vb.to(self.device)

        for i in range(len(b)):
            vb[:, i, :, :] = b[i]

        return s_0 *(f * torch.exp(- vb * d_1  * 1e-3) + (1 - f) * torch.exp(- vb * d_2 * 1e-3))

    def sigmoid_cons(self, params,cons):
        """
        params: parameter array
        cons: constraints cons[0]: lower bound cons[1]: upper bound
        """
        return cons[0]+torch.sigmoid(params)*(cons[1]-cons[0])

class myToyDataset(Dataset):
    '''
    Simulate a toy dataset to see if the training code can work. Data from random normal distribution
    '''
    def __init__(self, num_images, in_channels):
        super().__init__()
        self.data = torch.randn((num_images, in_channels, 128, 128))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        sample = {'image': image}

        return sample