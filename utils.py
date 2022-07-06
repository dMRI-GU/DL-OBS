from cmath import sqrt
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class load_data():
    '''
    This class load the data from the pointed directory
    '''

    def __init__(self, data_dir):
        '''
        data_dir: directory storing the data 'save_npy'
        '''
        super().__init__()
        self.data_dir = data_dir
        self.names = self.pat_names()
        self.pat_data = self.load()
    
    def load(self):
        '''
        Load the data from the data_dir
        '''
        pat_names = os.listdir(self.data_dir)
        pat_data = {}
        names = self.names

        for i, pat_d in enumerate(pat_names):

            'patd is DDR1.npy last four chars are discarded'
            name = pat_d[:-4]
            data_path = os.path.join(self.data_dir, pat_d)
            data = np.load(data_path, allow_pickle=True)[()]
            pat_data[names[i]] = data

        return pat_data
    
    def pat_names(self):
        '''
        Get the names of the patient data
        '''
        pat_names = os.listdir(self.data_dir)
        names = []

        for pat_d in pat_names:
            name = pat_d[:-4]
            names.append(name)

        return names

    def image_data(self, dir = 'S'):
        '''get the image data of the corresponding diffusion direction 
        (slices as batch size) 
        diffusion direction: I, S, M, P '''
        data_list = []        

        for pat_name in self.names:
            data = self.pat_data[pat_name]
            image_data = data['image_data']
            
            image_b0 = data['image_b0']
            image_b0[image_b0 == 0] = 1

            image_data_normalized = image_data / image_b0 
            
            diff_dir = data['diff_dir_arr']
            diff_dir = diff_dir[:, 0]
            mask_dir = (diff_dir == dir)
            image_dir = image_data_normalized[mask_dir]

            'Stack the image data regardless of the differences in the slices'
            data_list.append(image_dir)

        return np.concatenate(tuple(data_list), axis=1)

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
            images = images.to(torch.float32)

            out = self.net(images)
            s_0, d_1, d_2, f, sigma_g = self.parameter_maps(out_maps=out)
            M = self.rice_exp(s_0, d_1, d_2, f, sigma_g, b)
            val_losses += loss(M, images).item()
        
        return val_losses

    def rice_exp(self, s_0, d_1, d_2, f, sigma_g, b):
        'Get the expectation of the signal using denoised signal and std.'
        v = self.biexp(s_0, d_1, d_2, f, b)

        t = v / sigma_g
        res= sigma_g*(sqrt(torch.pi/8)*
                        ((2+t**2)*torch.special.i0e(t**2/4)+
                        t**2*torch.special.i1e(t**2/4)))

        return res.to(torch.float32)
    
    def parameter_maps(self, out_maps):
        'Get the parameter maps from the output'
        s_0, d_1 = out_maps[:, 0:1, :, :], out_maps[:, 1:2, :, :]
        d_2, f, sigma_g = out_maps[:, 2:3, :, :], out_maps[:, 3:4, :, :], out_maps[:, 4:5, :, :]
        
        s_0 = self.sigmoid_cons(s_0, 0.5, 1.5)
        d_1 = self.sigmoid_cons(d_1, 0, 4)
        d_2 = self.sigmoid_cons(d_2, 0, 0.1)
        f = self.sigmoid_cons(f, 0.1, 0.9)
        sigma_g = self.sigmoid_cons(sigma_g, 0.01, 0.2)

        return s_0, d_1, d_2, f, sigma_g

    def biexp(self, s_0, d_1, d_2, f, b):
        '''
        Reconstuct the denoised signal using the output parameter maps
        '''
        'vb (num of slices, b values, h, w)'
        b = b.view(1, len(b), 1, 1)

        return s_0 *(f * torch.exp(- b * d_1  * 1e-3) + (1 - f) * torch.exp(- b * d_2 * 1e-3))

    def sigmoid_cons(self, param, dmin, dmax):
        """
        params: parameter array
        cons: constraints cons[0]: lower bound cons[1]: upper bound
        """
        return dmin+torch.sigmoid(param)*(dmax-dmin)

class patientDataset(Dataset):
    '''
    Simulate a toy dataset to see if the training code can work. Data from random normal distribution
    '''
    def __init__(self, data):
        super(Dataset).__init__()
        self.data = torch.from_numpy(data) 
    
    def __len__(self):
        return self.data.shape[0] 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        sample = {'image': image}

        return sample
