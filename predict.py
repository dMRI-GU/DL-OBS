from model.unet_model import UNet
from model.unet_MultiDecoder import UNet_MultiDecoders
from torch.utils.data import DataLoader, random_split
from utils import load_data
from pathlib import Path
import os
import numpy as np
import torch
import argparse

result_path = Path('./results/')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--load', '-f', type=str, default='checkpoints/checkpoint_epoch3.pth',
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

    args = get_args()

    test_dir = 'test'
    load = load_data(test_dir)

    testX = load.image_data('M', normalize=True)
    testb0 = load.image_b0()
    'swap the dimension'
    testX = testX.transpose(1, 0, 2, 3)
    testX = load.crop_image(testX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testX = torch.from_numpy(testX)
    test_images = testX.to(device=device, dtype=torch.float32)
    test = torch.utils.data.TensorDataset(test_images)

    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8)

    b = torch.linspace(0, 3000, steps=21, device=device)
    b = b[1:]

    net = UNet_MultiDecoders(n_channels=test.shape[1], b=b, rice=True, bilinear=False)
    net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)

    total_loss = 0
    with torch.no_grad():
        for X in test_loader:
            mse = torch.nn.MSELoss()
            M, d_1, d_2, f, sigma = net(X)
            loss = mse(M, test_images)
            total_loss += loss.item()
        print(total_loss / len(test_loader))

    M, d_1, d_2, f, sigma = to_numpy(M, d_1, d_2, f, sigma)

    results = {'M.npy': M, 'd1.npy': d_1, 
                'd2.npy': d_2, 'f.npy': f, 'sigma_g.npy': sigma, 'b0': testb0}
    save_params(results)
