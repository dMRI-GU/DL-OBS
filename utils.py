from cmath import sqrt

import wandb
from torch.utils.data import Dataset
import torch
import os
from scipy import special
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from IPython import embed
from pytorch_msssim import MS_SSIM


class CustomLoss(nn.Module):

    """
    Custom loss function: :math:`L` = (1-SSIM) :math:`\cdot` MSEloss

    Example:

        >>>loss = CustomLoss()

        >>>loss_value = loss(predicted,target)
    """

    def __init__(self):
        super(CustomLoss, self).__init__()
        # For one dimensional channel, for example when inputting ADC map in loss function.
        self.ssim_loss = MS_SSIM(channel=1, win_size=5)
        self.ssim_loss2 = MS_SSIM(channel=20, win_size=5)#Default number of channels
        self.ssim_loss3 = MS_SSIM(channel=60, win_size=5)#If we use 3D in training
        self.mse_loss =  nn.L1Loss()#nn.MSELoss()
    def update_data_range(self, range):
        self.ssim_loss.data_range = range
    def forward(self, M,images, ssim_bool = False, only_ssim = False):
        """
      :param M: The predicted data/image
      :param images: The target data/image
      :param ssim_bool: If True, ssim is part of the loss function. Else only L1 is calculated
      :param only_ssim: If True, the loss function will only be = 1-ssim. L1 would not be included.
      """

        if M.shape[1]>1 and M.shape[1]<21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss2(M, images)
        elif M.shape[1]>21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss3(M, images)
        elif ssim_bool:
            loss_ssim = 1 - self.ssim_loss(M, images)
        else:
            loss_ssim = 1

        if not only_ssim:
            loss_mse = self.mse_loss(M,images)
        else:
            loss_mse = 1
        return loss_ssim * loss_mse

class post_processing():
    """
    This class include the post processing function to evaluate the trained model
    """
    def __init__(self):
        super().__init__()

    def evaluate(self, val_loader, net, rank, b, input_sigma: bool, ADC_loss,use_3D):
        """
        Here, validation data is used to monitor the training

        :param: val_loader: DataLoader for validation data
        :param: net: the neural network model
        :param: rank: Device-ID
        :param: b: Tensor of diffusion levels
        :param: input_sigma:Boolean, if a known noise map is input to the neural network.
        :param: ADC_loss: Boolean, if loss based on ADC maps is implemented.
        :param: use_3D: Boolean, if 3D is implemented.

        """
        criterion = CustomLoss()
        net.eval()
        val_losses = 0
        final_sigma = 0
        log_dict = {}
        save_dict = {}
        for i, (images, image_b0, sigma, scale_factor) in enumerate(val_loader):
            log_dict = {}
            save_dict = {}

            images = images.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 20 or 60 if use_3D , 200, 240)
            sigma = sigma.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 1, 200, 240)
            image_b0 = image_b0.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 1, 200, 240)
            scale_factor = scale_factor.to(rank, dtype=torch.float32, non_blocking=True)#(n_batches,)
            M, param_dict = net(images,b,image_b0, sigma, scale_factor)
            # M: (n_batches, 20 or 60 if use_3D, 200, 240)

            M = M * scale_factor.view(-1, 1, 1, 1)
            images = images * scale_factor.view(-1, 1, 1, 1)

            if ADC_loss:
                if use_3D:
                    slicing = [[slice(0, 1), slice(9, 10)],#To calculate ADC between b100 and b1000
                               [slice(20, 21), slice(29, 30)],
                               [slice(40, 41), slice(49, 50)]]
                else:
                    slicing = [[slice(0, 1), slice(9, 10)]]# Array[slice(0,1)] same as [Array[0]]
                ADC_avg_M = torch.zeros(size=(len(slicing), M.shape[0], 1, *M.shape[-2:]), device=rank)# len(slicing)=3 if 3D else = 1
                ADC_avg_images = torch.zeros(size=(len(slicing), images.shape[0], 1, *images.shape[-2:]), device=rank)

                for diff_index, sl in enumerate(slicing):
                    im100 = images[:, sl[0]]
                    im1000 = images[:, sl[1]]
                    im100 = im100 + torch.tensor(0.0001, device=im100.device)
                    im1000 = im1000 + torch.tensor(0.0001, device=im100.device)
                    ADC_avg_images[diff_index] = -torch.log(im1000 / im100) / (1000 - 100)

                    M100 = M[:, sl[0]]
                    M1000 = M[:, sl[1]]
                    M100 = M100 + torch.tensor(0.0001, device=im100.device)
                    M1000 = M1000 + torch.tensor(0.0001, device=im100.device)
                    ADC_avg_M[diff_index] = -torch.log(M1000 / M100) / (2000 - 100)
                ADC_avg_images = torch.mean(ADC_avg_images, dim=0)
                ADC_avg_M = torch.mean(ADC_avg_M, dim=0)
            ADC_loss_val = 1

            if ADC_loss:
                criterion.update_data_range(torch.max(ADC_avg_images))
                loss = 9 * 1000 * criterion(ADC_avg_M, ADC_avg_images, ssim_bool=False)
                # The 9 is arbitrary, depending on how much importance is given to the ADC loss relative to criterion(M,images)
                # The 1000 is for correct unit

                ADC_loss_val = loss.item()
                criterion.update_data_range(torch.max(images))
                loss += criterion(M, images, ssim_bool=True)

            else:
                criterion.update_data_range(torch.max(images))
                loss = criterion(M, images, ssim_bool=True)




            loss_value = torch.tensor(loss.item())

            if i == len(val_loader) - 1:#Last batch
                first_indices = [param_dict['names'].index(val) for val in dict.fromkeys(param_dict['names'])]
                #returns parameter names, e.g. ['d1','d2','f']

                for inde in first_indices:
                    res = param_dict['parameters'][inde][0]#Parameter array
                    key = param_dict['names'][inde]#Parameter name
                    res = res.cpu().detach().numpy()
                    res_min = np.min(res)
                    res_max = np.max(res)
                    log_dict.update({#For logging
                                     f'{key}_min': res_min,
                                     f'{key}_max': res_max,
                                     })
                    save_dict.update({f'{key}': res / res_max})#For saving the array as image

                if input_sigma:#If a known noise map is input
                    final_sigma = sigma[0,0,:,:]#The known noise map
                else:
                    final_sigma  =param_dict['sigma'][0,0,:,:]#Noise map from neural network


            val_losses += loss_value
                                                                #For saving the predicted and target image
        return val_losses/len(val_loader), log_dict, save_dict, M[0, 9, :, :], images[0, 9, :, :], final_sigma

class patientDataset(Dataset):
    '''
    wrap the patient numpy data to be dealt by the dataloader
    '''

    def __init__(self, data_dir, input_sigma: bool,use_3D:bool,fitting_model: str, transform=None, normalize = True, custom_list = None, crop=True, model_unetr = False):
        super(Dataset).__init__()
        self.data_dir = data_dir #Path to the diffusion data from patients
        self.transform = transform #Optional transformations to data
        self.use_3D = use_3D #Boolean, if 3D is implemented
        self.num_slices = 22
        self.num_direction = 3
        self.input_sigma = input_sigma #Boolean, if a known noise mpa is input to the neural network model
        self.crop = crop #If images are to be cropped before loaded during training
        #self.model_unetr = model_unetr
        self.fitting_model= fitting_model #Name of fitting model applied.

        # Must not include ToTensor()
        if custom_list is not None:#If we have a custom list of patients, then only those patients are included in dataset
            self.patients = custom_list
        else:
            self.patients = os.listdir(data_dir)#Else all patients in that folder are included
        self.data = self.load_npy_files_from_dir(data_directory= self.data_dir, patient_list=  self.patients)#Load all data. It gets saved to RAM
        print(len(self.data))
        self.normalize = normalize #Boolean, if we want to normalize data
        self.names = self.pat_names()

    def pat_names(self):
        """
        Save the patients' name in a list. e.g. ['pat1', 'pat2', ..., ...]
        """
        return [pat_d[:-4] for pat_d in os.listdir(self.data_dir)]


    def load_npy_files_from_dir(self, data_directory, patient_list):
        files = [f for f in os.listdir(data_directory) if f.endswith('.npy') and f in patient_list]  # List all .npy files
        data = []

        for file in files:
            file_path = os.path.join(data_directory, file)
            np_array = np.load(file_path, allow_pickle=True)[()]  # Load the .npy file
            im = np_array['image']['3Dsig']#The diffusion images
            b0 = np_array['image_b0']#b0-image
            result_biexp = np_array['result_biexp']#array with parameters from OBSIDIAN, e.g ['d1','d2','f','S0','sigma','nstep']
            result_kurtosis = np_array['result_kurtosis']
            result_gamma = np_array['result_gamma']
            data.append([im,b0,result_biexp,result_kurtosis,result_gamma])

        return data
    def __len__(self):
        """each data file consist of 22 slices, each slice acquired in three diffusion directions and in each direction diffusion weighted images at 20 b-values\n
            When using 3D, the total number of data samples decreases by 3-fold.

        """
        return len(self.patients)*self.num_slices*self.num_direction if not self.use_3D else len(self.patients)*self.num_slices
    
    def __getitem__(self, idx):
        # each time read on sample
        if torch.is_tensor(idx):
            idx = idx.tolist()
        direction_indice = idx//(self.num_slices*len(self.patients))
        pats_indice = idx // (self.num_slices*self.num_direction) if not self.use_3D else idx // (self.num_slices)
        slice_indice = idx % self.num_slices

        imgs,b0_data, sigma, factor = self.image_data(self.data[pats_indice], slice_indice, direction_indice,self.input_sigma, normalize=self.normalize, crop=self.crop)
        
        if self.transform:
            imgs = self.transform(imgs)

        return imgs,b0_data, sigma, factor#diffusion data, b0-image, noise map, scaling factor.

    def image_data(self, data, slice_idx, dir: int, input_sigma: bool, normalize=True, crop=True):
        """
        Get the image data of the corresponding diffusion direction (slices as batch size)

        :param: pat_path: The path to the directory of the patient data
        :param: slice_idx: The index of the slice
        :param: dir: Diffusion direction: 0,1,2
        :param: normalize: Boolean, if the image data is to be normalized by the max value of the diffusion images
        :param: crop: Boolean, if cropping the irrelevant background

        """

        idx = slice_idx

        # image_data - (num_slices,num_diffsuion_direction , H, W)
        image_data = data[0][idx, :, :, :]

        # image_b0 - (num_slices, H, W)
        image_b0 = data[1][idx, :, :]

        image_data = image_data.astype('float32')
        image_b0 = image_b0.astype('float32')
        image_data = torch.from_numpy(image_data)
        image_b0 = torch.from_numpy(image_b0)

        if self.fitting_model == 'biexp':
            fit_index = 2
        elif self.fitting_model == 'kurtosis':
            fit_index = 3
        elif self.fitting_model == 'gamma':
            fit_index = 4

        else: assert False, 'Not correct fitting model name'


        if input_sigma:
            sigma = data[fit_index][idx, :, :, -2]#Take noise map from OBSIDIAN
            sigma = sigma.astype('float32')
            sigma = torch.from_numpy(sigma)
        else:
            sigma = torch.tensor([1])

        if not self.use_3D:
            #If no 3D, then get data from one diffusion direction
            if dir == 0:
                image_data = image_data[0:20, :, :]
            elif dir == 1:
                image_data = image_data[20:40, :, :]

            elif dir == 2:
                image_data = image_data[40:60, :, :]

            else:
                print('ERROR: dir index is not 0,1 or 2')


        factor = torch.max(image_data)

        image_data = image_data / factor#Normalization

        print(image_b0.shape)
        image_b0 = image_b0.unsqueeze(dim=0)#(H,W) -> (1,H,W)
        print(image_b0.shape)

        if input_sigma:
            sigma = sigma / factor
            sigma = sigma.unsqueeze(dim=0)#(H,W) -> (1,H,W)

        # crop the redundant pixels
        if crop:
            image_data = self.crop_image(image_data)
            if input_sigma:
                sigma = self.crop_image(sigma)
            image_b0 = self.crop_image(image_b0)

        return image_data, image_b0, sigma, factor

    def crop_image(self, images):
        """
        (20, H, W)
        """
        #if self.model_unetr: return images[:, 16:-16, :]
        return images[:, 20:-20, :]


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