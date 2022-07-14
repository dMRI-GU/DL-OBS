from cmath import sqrt
from torch.utils.data import Dataset
import torch
import os
from scipy import special
import numpy as np
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class load_data():
    """
    This class load the data from the pointed directory
    """

    def __init__(self, data_dir):
        """
        data_dir: directory storing the data 'save_npy'
        """
        super().__init__()
        self.data_dir = data_dir
        self.names = self.pat_names()
        self.pat_data = self.load()
    
    def load(self):
        """
        Load the data from the data_dir
        """
        pat_names = os.listdir(self.data_dir)
        pat_data = {}

        for i, pat_d in enumerate(pat_names):

            'patd is DDR1.npy last four chars are discarded'
            name = pat_d[:-4]
            data_path = os.path.join(self.data_dir, pat_d)
            data = np.load(data_path, allow_pickle=True)[()]
            pat_data[name] = data

        return pat_data
    
    def pat_names(self):
        """
        """
        pat_names = os.listdir(self.data_dir)
        names = []

        for pat_d in pat_names:
            name = pat_d[:-4]
            names.append(name)

        return names

    def image_data(self, dir = 'M', normalize=True):
        """
        get the image data of the corresponding diffusion direction 
        (slices as batch size) 
        diffusion direction: I, S, M, P 
        """
        data_list = []        

        for pat_name in self.names:
            data = self.pat_data[pat_name]
            image_data = data['image_data']
            
            image_b0 = data['image_b0']
            if normalize:
                image_b0[image_b0 == 0] = 1
            else:
                image_b0 = 1

            image_data_normalized = image_data / image_b0 
            
            diff_dir = data['diff_dir_arr']
            diff_dir = diff_dir[:, 0]
            mask_dir = (diff_dir == dir)
            image_dir = image_data_normalized[mask_dir]

            'Stack the image data regardless of the differences in the slices'
            data_list.append(image_dir)

        return np.concatenate(tuple(data_list), axis=1)
    
    def crop_image(self, images):
        """
        (num_slices, 20, H, W)
        """
        return images[:, :, 20:-20, :]


class post_processing():
    """
    This class include the post processing function to evaluate the trained model
    """
    def __init__(self):

        super().__init__()
    
    def evaluate(self, val_loader, b, net, device):
        """
        evlaute the performance of network 
        """
        loss = torch.nn.MSELoss()
        net.eval()
        val_losses = 0

        params_val = dict()
        
        for batch in val_loader:
            images = batch['image']

            images = images.to(device=device, dtype=torch.float32)
            out, sigma = net(images)
            d_1, d_2, f, sigma_g = self.parameter_maps(out_maps=out, sigma_g=sigma)
            params_val = {'d_1':d_1, 'd_2':d_2, 'f':f, 'sigma_g':sigma}
            
            v = self.biexp(d_1, d_2, f, b)
            M = self.rice_exp(v, sigma_g)
            loss_value = loss(M, images).item()  
            val_losses += loss_value

        return val_losses, params_val, M[0, 0, :, :], images[0, 0, :, :]

    def rice_exp(self, v, sigma_g):
        """
        """

        t = v / sigma_g
        res= sigma_g*(sqrt(torch.pi/8)*
                        ((2+t**2)*torch.special.i0e(t**2/4)+
                        t**2*torch.special.i1e(t**2/4)))
        res = res.to(torch.float32)

        return res

    def parameter_maps(self, out_maps, sigma_g):
        """
        Get the parameter maps from the output
        """
        d_1, d_2 = out_maps[:, 0:1, :, :], out_maps[:, 1:2, :, :]
        f = out_maps[:, 2:3, :, :]
       
        d_1 = self.sigmoid_cons(d_1, 1.9, 2.6)
        d_2 = self.sigmoid_cons(d_2, 0.05, 0.7)
        f = self.sigmoid_cons(f, 0.3, 1.0)
        sigma_g = self.sigmoid_cons(sigma_g, 0, 12)

        return d_1, d_2, f, sigma_g

    def biexp(self, d_1, d_2, f, b):
        """
        """
        
        'vb (num of slices, b values, h, w)'
        b = b.view(1, len(b), 1, 1)

        return f * torch.exp(- b * d_1  * 1e-3) + (1 - f) * torch.exp(- b * d_2 * 1e-3)

    def sigmoid_cons(self, param, dmin, dmax):
        """
        """
        return dmin+torch.sigmoid(param)*(dmax-dmin)

class simulateDataset():
    """
    Simulate the data, the s_0 is normailized to 1
    """
    def __init__(self, num_images, sigma_low=5, sigma_high=40):
        self.d_1 = np.random.uniform(low=2, high=2.4, size=(num_images, 1, 240, 240))
        self.d_2 = np.random.uniform(low=0.1, high=0.5, size=(num_images, 1, 240, 240))
        self.f = np.random.uniform(low=0.5, high=0.9, size=(num_images, 1, 240, 240))
        self.b = np.linspace(0, 3000, 21)
        self.sigma_g = np.random.uniform(sigma_low,sigma_high,(num_images, 1, 240, 240))
        self.s_0 = 1

    def biexp(self):
        """
        Biexp model to get the denoised signal
        """

        b = self.b.reshape(1, len(self.b), 1, 1)
        v = self.f * np.exp(- b * self.d_1 * 1e-3) + (1 - self.f) * np.exp(- b * self.d_2 * 1e-3)

        'normalized the s_o to 1 and discard the first term which is s_0'
        return v[:, 1:, :, :] / v[:, 0:1, :, :]

    def rice_exp(self, v):
        """
        Get the reconstructed the rician exponential model
        """

        t = v / self.sigma_g
        res = self.sigma_g*(sqrt(np.pi/8)*
                        ((2+t**2)*special.i0e(t**2/4)+
                        t**2*special.i1e(t**2/4)))

        return res

class patientDataset(Dataset):
    '''
    wrap the patient numpy data to be dealt by the dataloader
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
        
