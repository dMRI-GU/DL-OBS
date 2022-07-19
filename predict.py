from model.unet_model import UNet
from utils import load_data
from pathlib import Path
import os
import numpy as np
import torch
import argparse

result_path = Path('./results/')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--load', '-f', type=str, default='checkpoints/checkpoint_epoch2.pth',
                        help='Load the model to test the result')

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

def save_params(**kwargs):
    """
    save the parameters maps and M as numpy array
    """
    Path(result_path).mkdir(parents=True, exist_ok=True)
    for key, res in kwargs.items():
        np.save(os.path.join(result_path, key), res)

if __name__ == '__main__':

    """
    This file load the trained model and run it on one patient, and 
    saves the result in ./results/
    """

    args = get_args()

    test_dir = 'test'
    load = load_data(test_dir)

    testX = load.image_data('M', normalize=True)
    'swap the dimension'
    testX = testX.transpose(1, 0, 2, 3)
    testX = load.crop_image(testX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = torch.from_numpy(testX)
    test_images = test.to(device=device)

    b = torch.linspace(0, 3000, steps=21, device=device)
    b = b[1:]

    net = UNet(n_channels=test.shape[1], b_values=b, rice=True, bilinear=False)
    net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)
    
    M, d_1, d_2, f, sigma = net(test_images)

    M, d_1, d_2, f, sigma = to_numpy(M, d_1, d_2, f, sigma)

    results = {'M.npy': M, 'd1.npy': d_1, 
                'd2.npy': d_2, 'f.npy': f, 'sigma_g.npy': sigma}
                