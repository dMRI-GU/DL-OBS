from utils import load_data 
import numpy as np
data_dir = 'save_npy'
pre_ = load_data(data_dir) 

pat_data = pre_.image_data(dir = 'M')

print(pat_data.shape)
