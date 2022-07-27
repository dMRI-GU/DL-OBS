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
    
    def image_b0(self):
        """
        Get the b0 for all the patients
        """
        b0s = [pat['image_b0'] for pat in self.pat_data.values()]
     
        "b0 for normalization"
        return [ np.where(b0 == 0, 1, b0) for b0 in b0s]

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
    
    def evaluate(self, val_loader, b, net, device, step):
        """
        evlaute the performance of network 
        """
        loss = torch.nn.L1Loss()
        net.eval()
        val_losses = 0

        params_val = dict()
       
        with torch.no_grad():
            for batch in val_loader:
                images = batch

                images = images.to(device=device, dtype=torch.float32)
                M, d_1, d_2, f, sigma_g = net(images)
                params_val = {'d_1':d_1, 'd_2':d_2, 'f':f, 'sigma_g':sigma_g}

                mse_loss = loss(M, images).item() 
                loss_value = torch.tensor(mse_loss)
                val_losses += loss_value

        return val_losses/len(val_loader), params_val, M[0, 0, :, :], images[0, 0, :, :]

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

        return image
        
