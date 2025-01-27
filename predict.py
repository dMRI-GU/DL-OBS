from model.unet_model import UNet
from model.attention_unet import Atten_Unet
from model.res_attention_unet import Res_Atten_Unet
from model.unet_MultiDecoder import UNet_MultiDecoders
from torch.utils.data import DataLoader, random_split
from utils import pre_data, patientDataset
from pathlib import Path
import os
import numpy as np
import torch
import argparse
import wandb
import torch.nn as nn
from tqdm import tqdm
from IPython import embed
result_path = Path('../results/')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--load', '-f', type=str, default='../Saved models/atten_ssim_mul_mse_new_param_limits.pth',
                        help='Load the model to test the result')
    parser.add_argument('--custom_patient_list', '-clist', type=str, default='predictList.txt', help='Input path to txt file with patient names to be used.')


    return parser.parse_args()

def to_numpy(*argv):
    """
    Convert the parameters from tensor to numpy
    """
    params = []
    for arg in argv:
        assert torch.is_tensor(arg), 'This is not a tensor'
        params.append(arg.cpu().detach().numpy())
    return params

def save_params(results):
    """
    save the parameters maps and M as numpy array
    """
    Path(result_path).mkdir(parents=True, exist_ok=True)
    for key, res in results.items():
        np.save(os.path.join(result_path, key), res)

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

    test_dir = '/m2_data/mustafa/patientData/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    test = patientDataset(test_dir,  custom_list=predict_list)
    test_loader = DataLoader(test, batch_size=22, shuffle=False, num_workers=4)
    #test_b0 = test.pre.image_b0()

    # Initialize the b values [100, 200, 300, ..., 2000]
    b = torch.linspace(0, 2000, steps=21, device=device)
    b = b[1:]
    b = b.reshape(1, len(b), 1, 1)
    # Load the UNet model
    net = Atten_Unet(n_channels=20, rice=False,n_classes=3, bilinear=False)
    #net = UNet(n_channels=20, b=b, rice=False, bilinear=False)
    #net = Res_Atten_Unet(n_channels=20, b=b, rice=False, bilinear=False)

    #net = nn.DataParallel(net)
    checkpoint = torch.load(args.load, map_location=device, weights_only=True)
    modified_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    net.load_state_dict(modified_checkpoint)
    net.to(device=device)

    total_loss = 0
    net.eval()
    #experiment = wandb.init(project="ResultFromTraining")
    b0_image = None
    n = len(test)
    with torch.no_grad():

        with tqdm(total=n, unit='img') as pbar:
            for i, (images,image_b0,sigma,scale_factor) in enumerate(test_loader):
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                sigma = sigma.to(device=device, dtype=torch.float32, non_blocking=True)
                image_b0 = image_b0.to(device=device, dtype=torch.float32, non_blocking=True)
                scale_factor = scale_factor.to(device=device, dtype=torch.float32, non_blocking=True)
                b = b.to(device=device, dtype=torch.float32, non_blocking=True)
                mse = torch.nn.MSELoss()
                M, d_1, d_2, f = net(images,b,image_b0, sigma,scale_factor)
                loss = mse(M, images)
                total_loss += loss.item()
                b0_image = image_b0

                pbar.update(images.shape[0])
            #experiment.log({'prediction': wandb.Image(M[0, 15, :, :], caption=f'patient {i}'),
            #                'image': wandb.Image(images[0, 15, :, :], caption=f'patient {i}')})

    print("Test Loss: {}".format(total_loss / len(test_loader)))



    M, d_1, d_2, f, sigma,b0_image = to_numpy(M*scale_factor.view(-1,1,1,1), d_1, d_2, f, sigma*scale_factor.view(-1,1,1,1), b0_image)

    results = {'M.npy': M, 'd1.npy': d_1, 
                'd2.npy': d_2, 'f.npy': f, 'sigma_g.npy': sigma, 'b0.npy': b0_image, 'images.npy':(images*scale_factor.view(-1,1,1,1)).detach().cpu().numpy()}
    
    # save the physical parameters and denoised images
    save_params(results)
