from cmath import sqrt
from torch.utils.data import Dataset
import torch
import os
from scipy import special
import numpy as np
import torch.nn as nn
import torchvision

class pre_data():
    """
    This class load the data from the pointed directory
    :"""

    def __init__(self, data_dir):
        """
        data_dir: directory storing the data 'save_npy'
        """
        super().__init__()
        self.data_dir = data_dir
        self.names = self.pat_names()
    
    def load(self, pat_path):
        """
        Load the data from the patient_path
        """
        data_path = os.path.join(self.data_dir, pat_path)
        pat_data = np.load(data_path, allow_pickle=True)[()]

        return pat_data
    
    def pat_names(self):
        """
        Save the patients' name in a list. e.g. ['pat1', 'pat2', ..., ...]
        """
        return [pat_d[:-4] for pat_d in os.listdir(self.data_dir)]

    def image_data(self, pat_path, slice_idx, dir = 'M', normalize=True, crop=True):
        """
        Get the image data of the corresponding diffusion direction (slices as batch size) 
        
        INPUT:
        pat_path - string- the path to the directory of the patient data
        slice_idx - the index of the slice
        dir - string - diffusion direction: I, S, M, P 
        normalize - boolean - if the image data is normalzied by its corresponding b0
        crop - boolean - if cropping the irrelevant background

        return:
        images - torch array: (1, 20, h, w) 20 is the number of diffusion direction
        """
        
        data = self.load(pat_path)
        idx = slice_idx

        #The data is saved as a dictionary with keys 'image_data' and 'image_b0'
        
        # image_data - (num_diffsuion_direction, num_slices, H, W)
        image_data = data['image_data'][:, idx, :, :]

        # image_b0 - (num_slices, H, W)
        image_b0 = data['image_b0'][idx, :, :]
        
        if normalize:
            image_b0[image_b0 == 0] = 1
        else:
            image_b0 = 1
        
        # (num_diffusion_direction, h, w)
        image_data_normalized = image_data / image_b0 
       
        # diffusion direction (num_diffusion_directuon, )
        diff_dir = data['diff_dir_arr'][:, 0]
        
        # mask_dir: (num_diffusion_direction,)
        mask_dir = (diff_dir == dir)

        #image_dir - (selected_num_diffusion_direction, h, w)
        image_dir = image_data_normalized[mask_dir] 
        imgs = torch.from_numpy(image_dir)

        # crop the redundant pixels
        if crop:
            imgs = self.crop_image(imgs)

        return imgs
    
    def image_b0(self):
        """
        Get the b0 for all the patients
        """
        files = os.listdir(self.data_dir)
        
        if not isinstance(files, list):
            files = [files]

        data = [self.load(file) for file in files]
        b0s = [pat_data['image_b0'] for pat_data in data]
     
        #b0 for normalization
        return [np.where(b0 == 0, 1, b0) for b0 in b0s]

    def crop_image(self, images):
        """
        (20, H, W)
        """
        return images[:, 20:-20, :]

class post_processing():
    """
    This class include the post processing function to evaluate the trained model
    """
    def __init__(self):
        super().__init__()
    
    def evaluate(self, val_loader, net, device):
        """
        evlaute the performance of network 
        """
        loss = torch.nn.MSELoss()
        net.eval()
        val_losses = 0

        params_val = dict()
       
        for batch in val_loader:
            images = batch

            images = images.to(device=device, dtype=torch.float32)
            M, d1, d2, f, sigma_g = net(images)
            params_val = {'d1':d1, 'd2': d2, 'f': f, 'sigma_g':sigma_g}

            mse_loss = loss(M, images).item() 
            loss_value = torch.tensor(mse_loss)
            val_losses += loss_value

        return val_losses/len(val_loader), params_val, M[0, 0, :, :], images[0, 0, :, :]

class patientDataset(Dataset):
    '''
    wrap the patient numpy data to be dealt by the dataloader
    '''
    def __init__(self, data_dir, transform=None, num_slices=22, normalize = True):
        super(Dataset).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.num_slices = 22

        # Must not include ToTensor()!
        self.patients = os.listdir(data_dir)
        self.pre = pre_data(data_dir)
        self.normalize = normalize

    def __len__(self):
        """each data file consist of 22 slices"""
        return len(os.listdir(self.data_dir))*self.num_slices
    
    def __getitem__(self, idx):
        # each time read on sample
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pats_indice = idx // self.num_slices
        slice_indice = idx % self.num_slices

        imgs = self.pre.image_data(self.patients[pats_indice], slice_indice, normalize=self.normalize)
        
        if self.transform:
            imgs = self.transform(imgs)

        return imgs

def init_weights(model):
    for name, module in model.named_modules():
        # Apply He initialization to Conv2d layers with ReLU activations
        if isinstance(module, nn.Conv2d):
            if 'att' in name:  # Attention layers
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for Sigmoid layers
            else:  # Conv2d layers using ReLU
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

        # Apply Xavier initialization to BatchNorm layers
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)  # Set the weight of BatchNorm to 1
            nn.init.constant_(module.bias, 0)  # Set the bias of BatchNorm to 0

        # Apply Xavier initialization for Linear layers if any (you may not have any in your current structure)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)