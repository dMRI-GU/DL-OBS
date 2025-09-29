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
from itertools import product

class CustomLoss(nn.Module):

    """
    Custom loss function: :math:`L` = L1-loss

    Example:

        >>>loss = CustomLoss()

        >>>loss_value = loss(predicted,target)
    """

    def __init__(self):
        super(CustomLoss, self).__init__()
        # For one dimensional channel, for example when inputting ADC map in loss function.
        self.mse_loss =  nn.L1Loss()

    def forward(self, M,images):
        """
      :param M: The predicted data/image
      :param images: The target data/image
      """

        loss_mse = self.mse_loss(M,images)

        return  loss_mse

class post_processing():
    """
    This class include the post processing function to evaluate the trained model
    """
    def __init__(self):
        super().__init__()

    def evaluate(self, val_loader, net, rank, b, input_sigma: bool, estimate_sigma,use_3D,include_sigma_loss):
        """
        Here, validation data is used to monitor the training

        :param: val_loader: DataLoader for validation data
        :param: net: the neural network model
        :param: rank: Device-ID
        :param: b: Tensor of diffusion levels
        :param: input_sigma:Boolean, if a known noise map is input to the neural network.
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
            if 'estimated_sigma' in param_dict:
                predicted_sigma = param_dict['estimated_sigma']

            loss = criterion(M, images)
            if 'estimated_sigma' in param_dict and include_sigma_loss and input_sigma:
                sigma_loss = criterion(predicted_sigma, sigma * scale_factor.view(-1, 1, 1, 1))
            else:
                sigma_loss = 0
            loss += sigma_loss




            loss_value = torch.tensor(loss.item())

            if i == len(val_loader) - 1:#Last batch
                first_indices = [param_dict['names'].index(val) for val in dict.fromkeys(param_dict['names'])]
                #returns parameter names, e.g. ['d1','d2','f']

                for index in first_indices:
                    res = param_dict['parameters'][index][0]#Parameter array
                    key = param_dict['names'][index]#Parameter name
                    res = res.cpu().detach().numpy()
                    res_min = np.min(res)
                    res_max = np.max(res)
                    log_dict.update({#For logging
                                     f'{key}_minev': res_min,
                                     f'{key}_maxev': res_max,
                                     })
                    save_dict.update({f'{key}': res / res_max})#For saving the array as image

                if 'estimated_sigma' in param_dict:
                    estimated_sigma = param_dict['estimated_sigma'].cpu().detach().numpy()[0]
                    log_dict.update({  # For logging
                        f'EstimatedSigma_minev': estimated_sigma.min(),
                        f'EstimatedSigma_maxev': estimated_sigma.max(),
                    })
                    save_dict.update({f'EstimatedSigma': estimated_sigma / estimated_sigma.max()})
                if input_sigma:
                    input_sigma_image = sigma[0,0,:,:].cpu().detach().numpy()
                    log_dict.update({  # For logging
                        f'TrueSigma_minev': input_sigma_image.min() ,
                        f'TrueSigma_maxev':  input_sigma_image.max(),
                    })
                    save_dict.update({f'TrueSigma': input_sigma_image /  input_sigma_image.max()})
                if input_sigma and not estimate_sigma :#If a known noise map is input
                    final_sigma = sigma[0,0,:,:]#The known noise map
                else:
                    final_sigma  =param_dict['sigma'][0,0,:,:]#Noise map from neural network
                final_sigma = final_sigma/final_sigma.max()

            val_losses += loss_value
                                                                #For saving the predicted and target image
        return val_losses/len(val_loader), log_dict, save_dict, M[0, 9, :, :], images[0, 9, :, :], final_sigma

class patientDataset(Dataset):
    '''
    wrap the patient numpy data to be dealt by the dataloader
    '''

    def __init__(self, data_dir, input_sigma: bool,use_3D:bool,fitting_model: str, transform=None, normalize = True, custom_list = None, crop=True,min_image_shape = None, model_unetr = False):
        super(Dataset).__init__()
        self.min_image_shape = min_image_shape
        self.data_dir = data_dir #Path to the diffusion data from patients
        self.transform = transform #Optional transformations to data
        self.use_3D = use_3D #Boolean, if 3D is implemented
        self.num_direction = 3
        self.input_sigma = input_sigma #Boolean, if a known noise mpa is input to the neural network model
        self.crop = crop #If images are to be cropped before loaded during training
        self.fitting_model= fitting_model #Name of fitting model applied.

        # Must not include ToTensor()
        if custom_list is not None:#If we have a custom list of patients, then only those patients are included in dataset
            self.patients = custom_list
        else:
            self.patients = None#Else all patients in that folder are included
        self.data,self.file_paths = self.load_npy_files_from_dir(data_directory= self.data_dir, patient_list=  self.patients)#Load all data. It gets saved to RAM
        self.n_slice_list = []
        self.indexList = self.indexify(data = self.data)
        self.slice_dict = {os.path.basename(path):self.n_slice_list[i]  for i,path in enumerate(self.file_paths)}
        self.normalize = normalize #Boolean, if we want to normalize data
        self.names = self.pat_names(patient_list=  self.patients)

    def indexify(self,data):

        tuples  = []
        for patient_index,patient in enumerate(data):
            num_slices = data[patient_index][0].shape[0]
            self.n_slice_list.append(num_slices)
            temp =[(patient_index, b, c) for b, c in product(range(num_slices), range(3))]
            tuples += sorted(temp, key=lambda x: (x[2], x[0], x[1]))
        return tuples

    def pat_names(self,patient_list):
        """
        Save the patients' name in a list. e.g. ['pat1', 'pat2', ..., ...]
        """
        data_directory = self.data_dir.split(",")
        if isinstance(data_directory,list):
            names = []
            for directory in data_directory:
                names +=  [ f.split('.')[0].split('_')[0] for f in os.listdir(directory) if f.endswith('.npy') and (patient_list is None or f in patient_list )]
        else:
            assert False, 'Something went wrong with dataset path'

        return names


    def load_npy_files_from_dir(self, data_directory, patient_list):
        data_directory = data_directory.split(",")

        if isinstance(data_directory,list):
            files = []
            for directory in data_directory:
                files +=  [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith('.npy') and (patient_list is None or f in patient_list )]
        else:
            assert False, 'Something went wrong with dataset path'

        data = []
        min_image_shape = [512,512]
        for file in files:
            np_array = np.load(file, allow_pickle=True)[()]  # Load the .npy file

            im = np_array['images']#The diffusion images
            b0 = np_array['image_b0']#b0-image
            result_biexp = np_array['result_biexp']#array with parameters from OBSIDIAN, e.g ['d1','d2','f','S0','sigma','nstep']
            result_kurtosis = np_array['result_kurtosis']
            result_gamma = np_array['result_gamma']
            data.append([im, b0, result_biexp, result_kurtosis, result_gamma])



            image_shape = im.shape[-2:]
            if image_shape[0]<min_image_shape[0]: min_image_shape[0] = image_shape[0]
            if image_shape[1]<min_image_shape[1]: min_image_shape[1] = image_shape[1]

        if self.min_image_shape is None: self.min_image_shape = min_image_shape
        return data,files
    def __len__(self):
        """each data file consist of 22 slices, each slice acquired in three diffusion directions and in each direction diffusion weighted images at 20 b-values\n
            When using 3D, the total number of data samples decreases by 3-fold.

        """
        return len(self.indexList) if not self.use_3D else len(self.indexList)//self.num_direction

    def __getitem__(self, idx):
        # each time read on sample
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pats_indice, slice_indice,direction_indice = self.indexList[idx]


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

        image_data = data[0][idx, :, :, :]

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

        image_b0 = image_b0.unsqueeze(dim=0)#(H,W) -> (1,H,W)

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
        target_h,target_w = self.min_image_shape
        _,h,w  = images.shape
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        return images[:,20:-20]

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
