import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Plot the M')
    parser.add_argument('--Directory', '-d', type=str, default='./results', help='Enter the folder storing the M.npy', 
                        dest='D')

    return parser.parse_args()

def plot(param, dir, mess):
    Path(dir).mkdir(parents=True, exist_ok=True)
    num_images = param.shape[0]

    if 'b' in mess:
        counts = np.linspace(0, 3000, 21)[1:]
        counts.astype(int)
    else:
        counts = np.arange(num_images) + 1

    for i in range(num_images):
        img = param[i]
        fig, ax = plt.subplots(1, 1,  figsize=(18,5))
        ax.imshow(img, cmap='gray', interpolation='none')
        plt.axis('off')
        fig.savefig(os.path.join(dir, f'{mess}_{counts[i]}.png'))
        plt.close()
    return

if __name__ == '__main__':
    args = get_args()

    data_dir = args.D
    
    'd1, d2, f and sigma_g are (22, 1, 200, 240)'
    d1 = np.load(os.path.join(data_dir, 'd1.npy'))
    d2 = np.load(os.path.join(data_dir, 'd2.npy'))
    f = np.load(os.path.join(data_dir, 'f.npy'))
    sigma_g = np.load(os.path.join(data_dir, 'sigma_g.npy'))

    results = {'d1': d1, 'd2': d2, 'f': f, 'sigma_g': sigma_g}


    'the b0 is (22, 1, 200, 240)'
    b0 = np.load(os.path.join(data_dir, 'b0.npy')).transpose(1, 0, 2, 3)

    # crop the image
    b0 = b0[:, :, 20:-20, :]

    bval = np.linspace(1, 3000, 21)[1:]
    bval = bval.reshape(1, len(bval), 1, 1)

    denoise_signal = (f*np.exp(-bval*d1*1e-3) +
                (1-f)*np.exp(-bval*d2*1e-3)) * b0

    denoise_slice1 = denoise_signal[0]

    [plot(np.squeeze(value, axis=1), os.path.join(data_dir, key), key) for key, value in results.items()]
    plot(denoise_slice1, os.path.join(data_dir, 'd_sl1'), 'b')
