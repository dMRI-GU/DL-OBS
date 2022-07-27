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

if __name__ == '__main__':
    args = get_args()

    M_dir = args.D
    
    'M - (22, 20, 200, 240)'
    M = np.load(os.path.join(M_dir, 'M.npy'))
    
    'the stored b0 is (1, 22, 200, 240)'
    b0 = np.load(os.path.join(M_dir, 'b0.npy')).transpose(1, 0, 2, 3)

    print(M.shape)

    'M - (22, 20, 200, 240)'
    M = M * b0[:, :, 20:-20, :]

    M_folder = os.path.join(M_dir, 'M')
    Path(M_folder).mkdir(parents=True, exist_ok=True)

    b = np.linspace(0, 3000, 21)[1:]

    M_slice1 = M[0]

    for i in range(20):
        fig, ax = plt.subplots(1, 1,  figsize=(18,5))
        img = M_slice1[i]
        ax.imshow(img, cmap='gray', interpolation='none')
        plt.axis('off')
        fig.savefig(os.path.join(M_folder, f'M_slice1_b{b[i]}.png'))
        plt.close()