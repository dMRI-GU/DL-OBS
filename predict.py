from pytorch_msssim import MS_SSIM
from model.unet_model import UNet
from model.attention_unet import Atten_Unet
from model.res_attention_unet import Res_Atten_Unet
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



def in_docker():
    # Check for /.dockerenv (standard Docker flag)
    if os.path.exists('/.dockerenv'):
        return True
    # Check cgroup info for Docker or container runtime
    try:
        with open('/proc/1/cgroup', 'rt') as f: return any('docker' in line or 'kubepods' in line or 'containerd' in line for line in f)
    except Exception:
        return False


result_path = Path('/write/your/path/')  if not in_docker() else Path('/app/result')
dataset_path1 = '/write/your/path/'  if not in_docker() else '/host-testset1/'
dataset_path2 = '/write/your/path/' if not in_docker() else '/host-testset2/'


class CustomLoss(nn.Module):

    """
    Custom loss function: :math:`L` = (1-SSIM) :math:`\cdot` MSEloss

    Example:

        >>>loss = CustomLoss()

        >>>loss_value = loss(predicted,target)
    """


    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ssim_loss = MS_SSIM(channel=1, win_size=5)
        self.ssim_loss2 = MS_SSIM(channel=20, win_size=5)
        self.ssim_loss3 = MS_SSIM(channel=60, win_size=5)
        self.mse_loss =  nn.L1Loss()#nn.MSELoss()
    def update_data_range(self, range):
        self.ssim_loss.data_range = range
    def forward(self, M,images, ssim_bool = False, only_ssim = False):
        if M.shape[1]>1 and M.shape[1]<21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss2(M, images)
        elif M.shape[1]>21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss3(M, images)
        elif ssim_bool:
            loss_ssim = 1 - self.ssim_loss(M, images)
        else:
            loss_ssim = 1

        if not only_ssim:
            loss_mse = self.mse_loss(M, images)
        else:
            loss_mse = 1
        return loss_ssim * loss_mse
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--load', '-f', type=str, default='/write/your/path/checkpoints/main_method',
                        help='Specify folder path to the neural network models to be used for inference')
    parser.add_argument('--custom_patient_list', '-clist', type=str, default='predictList.txt', help='Input path to txt file with patient names to be used for inference.')
    parser.add_argument('--rice', '-rice', action='store_true',help='Use this flag if you want to add Rician bias during inference')
    parser.add_argument('--filter', '-filter',  type=str, default='', help='Filter neural network models for inference , for example -filter attention_unet.')
    parser.add_argument('--test_data_directory', '-dir',  type=str, default=f'{dataset_path1},{dataset_path2}', help='Path to the test data.')
    parser.add_argument('--use_3D', '-3d',  action = 'store_true', help='If 3D')
    parser.add_argument('--learn_sigma_scaling', '-ss', action = 'store_true', help='Pass argument if AI was allowed to learn scaling sigma')
    parser.add_argument('--input_sigma', '-s',  action = 'store_true', help='Pass argument If a known noise map was inputted.')
    parser.add_argument('--estimate_S0', '-ests0', action = 'store_true', help='Pass argument if allowed AI to estimate S0-image')
    parser.add_argument('--feed_sigma', '-fs', action = 'store_true', help='Pass argument if feeding sigma map to AI. Input sigma has to be true')
    parser.add_argument('--estimate_sigma', '-estsigma', action = 'store_true', help='Pass argument if allowing for AI to estimate noise map')
    parser.add_argument('--use_true_sigma', '-uts', action = 'store_true',help='Pass argument if using noise map from OBSIDIAN in rice calculation')

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
            if torch.is_tensor(arg):
                params.append(arg.cpu().detach().numpy())
            elif isinstance(arg,list):
                params.append(np.array(arg))
            else:
                assert False, 'This is not a Tensor or a list'
    return params

def save_params(result_dict,main_folder:str,model_folder: str,fitting_folder: str ,patient_folder: str,run_number: str):
    """
    save the parameters maps and M as numpy array

    :param result_dict: Dictionery containing the parameter mapss to be saved.
    :param model_folder: Folder name given as the neural network model. Comes after *result_path*.
    :param fitting_folder: Folder name given as the fitting model. Comes after *model_folder*.
    :param patient_folder: Folder name given as the label to the diffusion data, e.g. a patient name. Comes after *fitting_folder*.
    :param run_number: Folder name given to separate separately trained models with the same settings, e.g. in a n-fold cross validation. Comes after *patient_folder*.

    """
    if args.rice:
        model_name_add = '_rician'#model name ends with rician if Rician bias is added
    else:
        model_name_add = ''
    complete_path = os.path.join(result_path,main_folder,model_folder+model_name_add,fitting_folder,patient_folder.split('.')[0],run_number)
    #example: /results/method_1/unet/biexp/run_1/

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

    if len(parts) < 4:  # Ensure there are at least three folders
        print('Error you .pth file should be in two nested folders.\n'
              'First folder is named after AI model used\n'
              'Second folder is named after fitting model used\n'
              'Third folder is named after the run number.')
        return None, None,None


    #With the structure of data saved, it returns model_name, fitting_name,run_number, file_name
    return parts[-4],parts[-3],parts[-2],path[last_slash + 1:last_dot] # Extract substring


if __name__ == '__main__':

    """
    This file load the trained model and run it on one patient, and 
    saves the result in *result_path*
    """
    UseBestCheckpoint = False
    Use200Checkpoint = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = get_args()
    assert not(args.feed_sigma and not args.input_sigma), 'Error: Argument input_sigma needs to be true if argument feed_sigma is passed'
    #Otherwise a tensor(1) is fed to neural network and not a proper noise map

    if args.custom_patient_list:
        with open(args.custom_patient_list, 'r') as file:
            # Read the entire file content and split by commas
            content = file.read().strip()  # Remove leading/trailing whitespace (if any)
            predict_list = content.split(',')#List of patients where inference is applied on

    test_dir = args.test_data_directory
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    folder_path = args.load
    main_folder_name = os.path.basename(os.path.normpath(folder_path))

    indexed_files = index_files(folder_path)
    assert len(indexed_files)>0, f'No files found at {folder_path}'
    print(folder_path)
    indexed_files = [f for f in indexed_files if args.filter in f]#If filtering is used, e.g. -filter '/unet' only runs inference on unet models
    if UseBestCheckpoint:
        indexed_files = [f for f in indexed_files if '_best' in f]

    else:
        indexed_files = [f for f in indexed_files if '_best' not in f]

    if Use200Checkpoint:
        indexed_files = [f for f in indexed_files if '200' in f]

    for f in indexed_files:
        print(f)



    for patient in predict_list:
        print(f'Running model for patient: {patient}::\n\n')
        for i,pth_file in enumerate(indexed_files):
            model_name, fitting_name,run_number, file_name = extract_file_name_folders(indexed_files[i]) # List of all files with their full paths
            print( model_name, fitting_name,run_number, file_name)
            # Load the test dataset
            test = patientDataset(test_dir,  custom_list=[patient], input_sigma=args.input_sigma, use_3D=args.use_3D, fitting_model=fitting_name, min_image_shape=[200,240])
            n_slice = test.slice_dict[patient]
            #Load all images of that patient
            if args.use_3D:
                batch_size = n_slice
            else:
                batch_size = n_slice*3

            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize the b values [100, 200, 300, ..., 2000]
            b = torch.linspace(0, 2000, steps=21, device=device)
            b = b[1:]#Exclude 0
            b = b.reshape(1, len(b), 1, 1)

            if args.use_3D:
                n_channels = 60#All three diffusion encoding directions are input
            else:
                n_channels = 20#Only one diffusion encoding directions is input

            print(f'We gave rice {args.rice}')
            # Load the neural network model
            if model_name == 'attention_unet':
                net = Atten_Unet(n_channels=n_channels, rice=args.rice, bilinear=False, input_sigma=args.input_sigma, fitting_model=fitting_name, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma)
            elif model_name == 'unet':
                net = UNet(n_channels=n_channels, rice=args.rice, bilinear=False, input_sigma=args.input_sigma,fitting_model=fitting_name, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma)
            elif model_name == 'res_atten_unet':
                net = Res_Atten_Unet(n_channels=n_channels, rice=args.rice, bilinear=False, input_sigma=args.input_sigma, fitting_model=fitting_name, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma)
            else:
                assert False, f'Could not find type of network model, i got {model_name}'

            #Load in the trained neural network for inference
            checkpoint = torch.load(indexed_files[i], map_location=device, weights_only=True)
            print(f'loaded model weights from {indexed_files[i]}\n')

            #Needed as with net = nn.parallel.DistributedDataParallel(net) adds the 'module' in the layers' labels.
            modified_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

            net.load_state_dict(modified_checkpoint)
            net.to(device=device)

            net.eval()
            b0_image = None
            n = len(test)#number of images in a patient
            results = {}
            criterion = CustomLoss()
            with torch.no_grad():

                with tqdm(total=n, unit='img') as pbar:
                    for i, (images,image_b0,sigma,scale_factor) in enumerate(test_loader):
                        images = images.to(device=device, dtype=torch.float32, non_blocking=True)# (66, 20, 200, 240)
                        sigma = sigma.to(device=device, dtype=torch.float32, non_blocking=True)# (66, 1, 200, 240)
                        image_b0 = image_b0.to(device=device, dtype=torch.float32, non_blocking=True)# (66, 1, 200, 240)
                        scale_factor = scale_factor.to(device=device, dtype=torch.float32, non_blocking=True)# (66,)
                        b = b.to(device=device, dtype=torch.float32, non_blocking=True)# (1, 20, 1, 1)

                        mse = torch.nn.MSELoss()
                        M, param_dict = net(images,b,image_b0, sigma,scale_factor)

                        M = M * scale_factor.view(-1, 1, 1, 1)
                        images = images * scale_factor.view(-1, 1, 1, 1)
                        b0_image = image_b0
                        criterion.update_data_range(torch.max(images))
                        loss = criterion(M, images, ssim_bool=True)
                        loss_np = np.array(loss.item())
                        M_np,param_dict_np = to_numpy(M, param_dict)
                        results.update(param_dict_np)
                        results.update({'M':M_np,'loss': loss_np})
                        pbar.update(images.shape[0])
                        save_params(result_dict= results,main_folder=main_folder_name, model_folder = model_name,fitting_folder  =fitting_name,patient_folder = patient, run_number = run_number )
                        print('Saved this run\n')