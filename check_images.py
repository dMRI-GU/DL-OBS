from utils import load_data
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'save_npy'

pre_ = load_data(data_dir) 

pat_data = pre_.image_data(dir = 'I')

pat_data = np.swapaxes(pat_data, 1, 0)
print(pat_data.shape)

'Get four slices of images'

images = pat_data[0:4]
for i, image in enumerate(images):
    print(np.sum(image == 1))

'''
Plot the images
'''
'''fig, ax = plt.subplots(1, len(pat_data),  figsize=(18,5))
for i, image in enumerate(pat_data):
    print(image.shape)
    ax[i].imshow(image, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
plt.show()'''
