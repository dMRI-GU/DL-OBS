from pytorch_msssim import MS_SSIM

from model.unet_model import UNet
from model.attention_unet import Atten_Unet
from model.res_attention_unet import Res_Atten_Unet
from model.unet_MultiDecoder import UNet_MultiDecoders
from torch.utils.data import DataLoader, random_split
from utils import patientDataset
from pathlib import Path
import os
import numpy as np
import torch
import argparse
import wandb
import torch.nn as nn
from tqdm import tqdm
from IPython import embed
result_path = Path('../results/L2loss/')

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ssim_loss = MS_SSIM(channel=20,win_size=5)
        self.mse_loss =  nn.MSELoss()#nn.L1Loss()
    def update_data_range(self, range):
        self.ssim_loss.data_range = range
    def forward(self, M,images):
        # Example: Mean squared error + L1 regularization
        print(f'inside: dtype M: {M.dtype} and images {images.dtype}')
        loss_ssim = 1 - self.ssim_loss(M.float(), images.float())
        loss_mse = self.mse_loss(M, images)
        return loss_ssim * loss_mse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--load', '-f', type=str, default='../checkpoints/cross_validation_l2',
                        help='Load the model to test the result')
    parser.add_argument('--custom_patient_list', '-clist', type=str, default='predictList.txt', help='Input path to txt file with patient names to be used.')
    parser.add_argument('--epoch_number', '-enum', type=str, default='30', help='Input epoch number to be used for prediction.')


    return parser.parse_args()

def to_numpy(*argv):
    """
    Convert the parameters from tensor to numpy. Also handles dictionaries containing tensors.
    """
    params = []
    for arg in argv:
        if isinstance(arg, dict):  # Check if the argument is a dictionary
            converted_dict = {key: to_numpy(value)[0] for key, value in arg.items()}
            params.append(converted_dict)
        else:
            # Ensure that the argument is a tensor
            assert torch.is_tensor(arg), 'This is not a tensor'
            params.append(arg.cpu().detach().numpy())
    return params

def save_params(result_dict,model_folder: str,fitting_folder: str ,patient_folder: str,run_number: str, file_name :str ):
    """
    save the parameters maps and M as numpy array
    """
    complete_path = os.path.join(result_path,model_folder,fitting_folder,patient_folder.split('_')[0],run_number)
    Path(complete_path).mkdir(parents=True, exist_ok=True)
    for key, res in result_dict.items():
        if torch.is_tensor(res):
            res = res.cpu().detach().numpy()  # Move tensor to CPU if it's on CUDA
            print(f'{key} is not on cpu. Moved to cpu')
        npy_file_name =key+'.npy'
        np.save(os.path.join(complete_path,npy_file_name ), res)
def index_files(path, file_list=None):
    """ Recursively index files inside folders and return a flat list of file paths. """
    if file_list is None:
        file_list = []  # Initialize the list on first call
    path = Path(path)
    if path.is_dir():
        for item in sorted(path.iterdir()):  # Sort for consistent ordering
            if item.is_file():
                file_list.append(str(item))  # Store the full file path
            elif item.is_dir():
                index_files(item, file_list)  # Recurse into subfolders
    return file_list


def extract_file_name_folders(path):
    last_dot = path.rfind(".")  # Find last occurrence of '.'
    last_slash = path.rfind("/")  # Find last occurrence of '/'

    if last_dot == -1 or last_slash == -1 or last_slash > last_dot:
        print('Error: Your something wrong with file naming och folder structure')
        return None,None,None  # Handle cases where conditions are not met

    path_norm = Path(path).resolve()  # Normalize the path
    parts = path_norm.parts  # Get all path components

    if len(parts) < 4:  # Ensure there are at least two folders
        print('Error you .pth file should be in two nested folders.\n'
              'First folder is named after AI model used\n'
              'Second folder is named after fitting model used.')
        return None, None,None



    return parts[-4],parts[-3],parts[-2],path[last_slash + 1:last_dot]  # Extract substring


if __name__ == '__main__':

    """
    This file load the trained model and run it on one patient, and 
    saves the result in ./results/
    """

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = get_args()
    if args.custom_patient_list:
        with open(args.custom_patient_list, 'r') as file:
            # Read the entire file content and split by commas
            content = file.read().strip()  # Remove leading/trailing whitespace (if any)
            predict_list = content.split(',')

    test_dir = '/m2_data/mustafa/nonTrainData/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example usage:
    folder_path = args.load
    indexed_files = index_files(folder_path)
    indexed_files = [f for f in indexed_files if args.epoch_number in f]
    for patient in predict_list:
        print(f'Running model for patient: {patient}::\n\n')
        for i,pth_file in enumerate(indexed_files):
            model_name, fitting_name,run_number, file_name = extract_file_name_folders(indexed_files[i]) # List of all files with their full paths

            # Load the test dataset
            test = patientDataset(test_dir,  custom_list=[patient], input_sigma=True)
            test_loader = DataLoader(test, batch_size=66, shuffle=False, num_workers=4)
            #test_b0 = test.pre.image_b0()

            # Initialize the b values [100, 200, 300, ..., 2000]
            b = torch.linspace(0, 2000, steps=21, device=device)
            b = b[1:]
            b = b.reshape(1, len(b), 1, 1)
            # Load the UNet model
            if model_name == 'attention_unet':
                net = Atten_Unet(n_channels=20, rice=False, bilinear=False, input_sigma=True, fitting_model=fitting_name)

            elif model_name == 'unet':
                net = UNet(n_channels=20, rice=False, bilinear=False, input_sigma=True,fitting_model=fitting_name)
            elif model_name == 'res_atten_unet':
                net = Res_Atten_Unet(n_channels=20, rice=False, bilinear=False, input_sigma=True, fitting_model=fitting_name)


            #net = UNet(n_channels=20, b=b, rice=False, bilinear=False)
            #net = Res_Atten_Unet(n_channels=20, b=b, rice=False, bilinear=False)

            #net = nn.DataParallel(net)
            checkpoint = torch.load(indexed_files[i], map_location=device, weights_only=True)
            print(f'loaded model weights from {indexed_files[i]}\n')
            modified_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

            net.load_state_dict(modified_checkpoint)
            net.to(device=device)

            net.eval()
            #experiment = wandb.init(project="ResultFromTraining")
            b0_image = None
            n = len(test)
            results = {}
            criterion = CustomLoss()
            with torch.no_grad():

                with tqdm(total=n, unit='img') as pbar:
                    for i, (images,image_b0,sigma,scale_factor) in enumerate(test_loader):
                        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                        sigma = sigma.to(device=device, dtype=torch.float32, non_blocking=True)
                        image_b0 = image_b0.to(device=device, dtype=torch.float32, non_blocking=True)
                        scale_factor = scale_factor.to(device=device, dtype=torch.float32, non_blocking=True)
                        b = b.to(device=device, dtype=torch.float32, non_blocking=True)
                        mse = torch.nn.MSELoss()
                        M, param_dict = net(images,b,image_b0, sigma,scale_factor)
                        M = M * scale_factor.view(-1, 1, 1, 1)
                        images = images * scale_factor.view(-1, 1, 1, 1)
                        print(f'dtype M: {M.dtype} and images {images.dtype}')
                        b0_image = image_b0
                        criterion.update_data_range(torch.max(images))
                        loss = criterion(M, images)
                        loss_np = np.array(loss.item())
                        M_np,param_dict_np = to_numpy(M, param_dict)
                        results.update(param_dict_np)
                        results.update({'M':M_np,'loss': loss_np})
                        pbar.update(images.shape[0])
                        save_params(result_dict= results, model_folder = model_name,fitting_folder  =fitting_name,patient_folder = patient, run_number = run_number,file_name = file_name )
                        print('Saved this run\n')