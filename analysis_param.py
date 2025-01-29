import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
matplotlib.use('TkAgg')
from math import sqrt
import torch
import pandas as pd
from tabulate import tabulate
import nibabel as nib

def rice_exp(v, sigma):
    """
    Add the rician bias
    """
    t = v / sigma
    res= sigma*(sqrt(torch.pi/8)*
                    ((2+t**2)*torch.special.i0e(t**2/4)+
                    t**2*torch.special.i1e(t**2/4)))
    res = res.to(torch.float32)
    return res

def bio_exp(d1, d2, f, b):
    """ivim model"""
    v = f*torch.exp(-b*d1*1e-3+1e-6) + (1-f)*torch.exp(-b*d2*1e-3+1e-6)

    return v


#0     0    0    0        0  0  0    "Clear Label"
#1   255    0    0        1  1  1    "Lesion_peri"
#2     0  255    0        1  1  1    "Lesion_cent"
#3     0    0  255        1  1  1    "Healthy_peri"
#4   255  255    0        1  1  1    "Healthy_cent"
#5     0  255  255        1  1  1    "Lesion_peri_certain"
#6   255    0  255        1  1  1    "Lesion_cent_certain"



current_index = 0
vindex = 0
base_path = '/mnt/mustafa/results/atten_ssim_mul_mse_new_param_limits/'
file = np.load('/mnt/mustafa/denoise-unet/test/AD56_biexp.npy', allow_pickle=True)[()]
roi_file = nib.load('/mnt/mustafa/denoise-unet/test/AD56_seg.nii.gz')
roi_file = np.asanyarray(roi_file.dataobj).transpose(2, 1, 0)
roi_file = roi_file[:,20:-20,:]
#find image

fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # 5x5 grid for 22 images
axes = axes.flatten()
# Plot each slice in a separate subplot
for i in range(22):
    axes[i].imshow(roi_file[i], cmap='gray')  # Display the i-th slice
    axes[i].axis('off')  # Hide axis labels


plt.tight_layout()
plt.show()
matplotlib.use('TkAgg')



true_sigma = torch.from_numpy(file['result']['3Dsig'][:,:,:,10])
true_d1 = torch.from_numpy(file['result']['3D'][:,:,:,0])
true_d2 = torch.from_numpy(file['result']['3D'][:,:,:,1])
true_f = torch.from_numpy(file['result']['3D'][:,:,:,2])
M = np.load(f'{base_path}M.npy',allow_pickle=True)
d1 = np.load(f'{base_path}d1.npy',allow_pickle=True)
d2 = np.load(f'{base_path}d2.npy',allow_pickle=True)
f = np.load(f'{base_path}f.npy',allow_pickle=True)

#.---------------



def updateTable(roi_image, M,d1,d2,f,true_image,true_d1, true_d2, true_f, current_index, update = False):
    regions = np.unique(roi_image)#roiimage (22,200,240)
    results = []
    for region in regions:
        if region == 0:
            continue  # Skip background or unlabeled regions if 0 is the background

        # Mask the region
        region_mask = roi_image == region

        # Calculate statistics for each image
        M_table = M[:,current_index,:,:]
        M_mean = np.mean(M_table[region_mask])
        M_std = np.std(M_table[region_mask])
        true_image_table= true_image[:,current_index,20:-20,:]
        image_mean = np.mean(true_image_table[region_mask])
        image_std = np.std(true_image_table[region_mask])

        d1_table = d1[:,0,:,:]
        d1_mean = np.mean(d1_table[region_mask])
        d1_std = np.std(d1_table[region_mask])

        true_d1_mean = np.mean(true_d1[region_mask])
        true_d1_std = np.std(true_d1[region_mask])

        d2_table = d2[:, 0, :, :]
        d2_mean = np.mean(d2_table[region_mask])
        d2_std = np.std(d2_table[region_mask])
        true_d2_mean = np.mean(true_d2[region_mask])
        true_d2_std = np.std(true_d2[region_mask])

        f_table = f[:, 0, :, :]
        f_mean = np.mean(f_table[region_mask])
        f_std = np.std(f_table[region_mask])
        true_f_mean = np.mean(true_f[region_mask])
        true_f_std = np.std(true_f[region_mask])


        # Append results
        results.append({
            'Region': region,
            'Predicted image': f'{M_mean:.3f}±{M_std:.3f}',
            'True image': f'{image_mean:.3f}±{image_std:.3f}',
            'Predicted D1': f'{d1_mean:.3f}±{d1_std:.3f}',
            'True D1': f'{true_d1_mean:.3f}±{true_d1_std:.3f}',
            'Predicted D2': f'{d2_mean:.3f}±{d2_std:.3f}',
            'True D2': f'{true_d2_mean:.3f}±{true_d2_std:.3f}',
            'Predicted f': f'{f_mean:.3f}±{f_std:.3f}',
            'True f': f'{true_f_mean:.3f}±{true_f_std:.3f}',

        })


    results_df = pd.DataFrame(results)
    # Print the table with borders
    if not update:
        print(tabulate(results_df, headers='keys', tablefmt="latex"))


M_data = M[vindex,current_index,:,:]
d1_data  = d1[vindex,0,:,:]
d2_data  = d2[vindex,0,:,:]
f_data  = f[vindex,0,:,:]
b0 = torch.tensor(file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim =1)



b = torch.linspace(0, 2000, steps=21)
b = b[1:]
b = b.reshape(1,len(b), 1, 1)


true_M = b0*bio_exp(true_d1.unsqueeze(dim = 1), true_d2.unsqueeze(dim = 1), true_f.unsqueeze(dim = 1), b)
#true_M = rice_exp(b0*v[current_index,:,:], true_sigma_data).numpy()
true_M_data = true_M[vindex,current_index,20:-20,:].numpy()
true_d1_data = true_d1[vindex,20:-20,:].numpy()
true_d2_data = true_d2[vindex,20:-20,:].numpy()
true_f_data = true_f[vindex,20:-20,:].numpy()
true_sigma_data = true_sigma[vindex,20:-20,:]

roi_image = roi_file
true_image = true_M.numpy()
updateTable(roi_image, M,d1,d2,f,true_image,true_d1[:,20:-20,:].numpy(), true_d2[:,20:-20,:].numpy(), true_f[:,20:-20,:].numpy(), current_index)

fig, ax = plt.subplots(4, 2, figsize=(10, 4))  # 1 row, 2 columns
fig2, ax2 = plt.subplots(4, 2, figsize=(10, 4))  # 1 row, 2 columns
ax_table = fig2.add_subplot(5, 1, 5)  # Combine the last two columns
ax_table.axis('off')

plt.show()
matplotlib.use('TkAgg')


cmap_range = [np.min(M_data), np.max(M_data)]
cmap_range1 = [np.min(true_M_data), np.max(true_M_data)]
cmap_range2 = [np.min(d1_data), np.max(d1_data)]
cmap_range3 = [np.min(true_d1_data), np.max(true_d1_data)]
cmap_range4 = [np.min(d2_data), np.max(d2_data)]
cmap_range5 = [np.min(true_d2_data), np.max(true_d2_data)]
cmap_range6 = [np.min(f_data), np.max(f_data)]
cmap_range7 = [np.min(true_f_data), np.max(true_f_data)]

# Plot data in the first subplot
Mdisplay = ax[0,0].imshow(M_data, vmin=cmap_range[0], vmax=cmap_range[1],cmap = 'gray')
ax[0,0].set_title(f'M[{vindex},{current_index},:,:]')

# Plot data in the second subplot
images_display = ax[0,1].imshow(true_M_data, vmin=cmap_range1[0], vmax=cmap_range1[1],cmap = 'gray')
ax[0,1].set_title(f'True image[{vindex},{current_index},:,:]')

d1_display = ax[1,0].imshow(d1_data, vmin=cmap_range2[0], vmax=cmap_range2[1], cmap ='gray')
ax[1,0].set_title(f'd1_data[{vindex},:,:]')

true_d1_display = ax[1,1].imshow(true_d1_data, vmin=cmap_range3[0], vmax=cmap_range3[1], cmap ='gray')
ax[1,1].set_title(f'True d1[{vindex},:,:]')

d2_display = ax[2,0].imshow(d2_data, vmin=cmap_range4[0], vmax=cmap_range4[1], cmap ='gray')
ax[2,0].set_title(f'd2_data[{vindex},:,:]')

true_d2_display = ax[2,1].imshow(true_d2_data, vmin=cmap_range5[0], vmax=cmap_range5[1], cmap ='gray')
ax[2,1].set_title(f'True d2[{vindex},:,:]')

f_display = ax[3,0].imshow(f_data, vmin=cmap_range6[0], vmax=cmap_range6[1], cmap ='gray')
ax[3,0].set_title(f'f_data[{vindex},:,:]')

true_f_display = ax[3,1].imshow(true_f_data, vmin=cmap_range7[0], vmax=cmap_range7[1], cmap ='gray')
ax[3,1].set_title(f'True f[{vindex},:,:]')


ax[0,0].axis('off')
ax[0,1].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[2,0].axis('off')
ax[2,1].axis('off')
ax[3,0].axis('off')
ax[3,1].axis('off')


cbar  = fig.colorbar(Mdisplay, ax=ax[0,0], label='Interactive colorbar')
cbar1  = fig.colorbar(images_display, ax=ax[0,1], label='Interactive colorbar')

cbar2  = fig.colorbar(d1_display, ax=ax[1,0], label='Interactive colorbar')
cbar3  = fig.colorbar(true_d1_display, ax=ax[1,1], label='Interactive colorbar')

cbar4  = fig.colorbar(d2_display, ax=ax[2,0], label='Interactive colorbar')
cbar5  = fig.colorbar(true_d2_display, ax=ax[2,1], label='Interactive colorbar')

cbar6  = fig.colorbar(f_display, ax=ax[3,0], label='Interactive colorbar')
cbar7  = fig.colorbar(true_f_display, ax=ax[3,1], label='Interactive colorbar')



###################



ax2[0,0].hist(M_data.reshape(-1), bins=100, color='blue')
ax2[0,0].set_title(f'M[{vindex},{current_index},:,:]')

# Plot data in the second subplot
ax2[0,1].hist(true_M_data.reshape(-1), bins=100, color='red')
ax2[0,1].set_title(f'True image[{vindex},{current_index},:,:]')

ax2[1,0].hist(d1_data.reshape(-1), bins=100, color='black')
ax2[1,0].set_title(f'd1[{vindex},:,:]')

ax2[1,1].hist(true_d1_data.reshape(-1), bins=100, color='black')
ax2[1,1].set_title(f'True d1[{vindex},:,:]')

ax2[2,0].hist(d2_data.reshape(-1), bins=100, color='black')
ax2[2,0].set_title(f'd2[{vindex},:,:]')

ax2[2,1].hist(true_d2_data.reshape(-1), bins=100, color='black')
ax2[2,1].set_title(f'True d2[{vindex},:,:]')

ax2[3,0].hist(f_data.reshape(-1), bins=100, color='black')
ax2[3,0].set_title(f'f[{vindex},:,:]')

ax2[3,1].hist(true_f_data.reshape(-1), bins=100, color='black')
ax2[3,1].set_title(f'True f[{vindex},:,:]')


# Adjust layout to prevent overlap
#plt.tight_layout()

# Show the plot
def on_key(event):
    global current_index
    global vindex
    global ax2
    global cmap_range
    global cmap_range1
    global cmap_range2
    global cmap_range3
    global cmap_range4
    global cmap_range5
    global cmap_range6
    global cmap_range7
    if event.key == 'right':  # Scroll forward
        current_index = (current_index + 1)%20  # Loop back to the start
    elif event.key == 'left':  # Scroll backward
        current_index = (current_index - 1)%20  # Loop to the end if needed
    elif event.key == 'up':  # Scroll forward
        vindex = (vindex + 1) % M.shape[0]  # Loop back to the start
    elif event.key == 'down':  # Scroll backward
        vindex = (vindex - 1) % M.shape[0]  # Loop back to the start

    M_data = M[vindex, current_index, :, :]
    d1_data = d1[vindex, 0, :, :]
    d2_data = d2[vindex, 0, :, :]
    f_data = f[vindex, 0, :, :]
    b0 = torch.tensor(file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)

    b = torch.linspace(0, 2000, steps=21)
    b = b[1:]
    b = b.reshape(1, len(b), 1, 1)

    true_M = b0 * bio_exp(true_d1.unsqueeze(dim=1), true_d2.unsqueeze(dim=1), true_f.unsqueeze(dim=1), b)
    # true_M = rice_exp(b0*v[current_index,:,:], true_sigma_data).numpy()
    true_M_data = true_M[vindex, current_index, 20:-20, :].numpy()
    true_d1_data = true_d1[vindex, 20:-20, :].numpy()
    true_d2_data = true_d2[vindex, 20:-20, :].numpy()
    true_f_data = true_f[vindex, 20:-20, :].numpy()
    true_sigma_data = true_sigma[vindex, 20:-20, :].numpy()

    roi_image = roi_file
    true_image = true_M.numpy()
    updateTable(roi_image, M, d1, d2, f, true_image, true_d1[:, 20:-20, :].numpy(), true_d2[:, 20:-20, :].numpy(),
                true_f[:, 20:-20, :].numpy(), current_index, update = True)

    Mdisplay.set_data(M_data)
    images_display.set_data(true_M_data)
    d1_display.set_data(d1_data)
    true_d1_display.set_data(true_d1_data)
    d2_display.set_data(d2_data)
    true_d2_display.set_data(true_d2_data)
    f_display.set_data(f_data)
    true_f_display.set_data(true_f_data)
    ax[0,0].set_title(f'M[{vindex},{current_index},:,:]')
    ax[0,1].set_title(f'True image[{vindex},{current_index},:,:]')
    ax[1,0].set_title(f'd1[{vindex},:,:]')
    ax[1,1].set_title(f'True d1[{vindex},:,:]')
    ax[2,0].set_title(f'd2[{vindex},:,:]')
    ax[2,1].set_title(f'True d2[{vindex},:,:]')
    ax[3,0].set_title(f'f[{vindex},:,:]')
    ax[3,1].set_title(f'True f[{vindex},:,:]')

    ax2[0,0].clear()
    ax2[0,1].clear()
    ax2[1,0].clear()
    ax2[1,1].clear()
    ax2[2,0].clear()
    ax2[2,1].clear()
    ax2[3,0].clear()
    ax2[3,1].clear()
    ax2[0,0].hist(M_data.reshape(-1), bins=100, color='blue')
    ax2[0,0].set_title(f'M[{vindex},{current_index},:,:]')

    ax2[0,1].hist(true_M_data.reshape(-1), bins=100, color='red')
    ax2[0,1].set_title(f'True image[{vindex},{current_index},:,:]')

    ax2[1,0].hist(d1_data.reshape(-1), bins=100, color='black')
    ax2[1,0].set_title(f'd1[{vindex},:,:]')

    ax2[1,1].hist(true_d1_data.reshape(-1), bins=100, color='black')
    ax2[1,1].set_title(f'True d1[{vindex},:,:]')

    ax2[2,0].hist(d2_data.reshape(-1), bins=100, color='black')
    ax2[2,0].set_title(f'd2[{vindex},:,:]')

    ax2[2, 1].hist(true_d2_data.reshape(-1), bins=100, color='black')
    ax2[2, 1].set_title(f'True d2[{vindex},:,:]')

    ax2[3, 0].hist(f_data.reshape(-1), bins=100, color='black')
    ax2[3, 0].set_title(f'f[{vindex},:,:]')

    ax2[3, 1].hist(true_f_data.reshape(-1), bins=100, color='black')
    ax2[3, 1].set_title(f'f[{vindex},:,:]')

    cmap_range = [np.min(M_data), np.max(M_data)]
    cmap_range1 = [np.min(true_M_data), np.max(true_M_data)]
    cmap_range2 = [np.min(d1_data), np.max(d1_data)]
    cmap_range3 = [np.min(true_d1_data), np.max(true_d1_data)]
    cmap_range4 = [np.min(d2_data), np.max(d2_data)]
    cmap_range5 = [np.min(true_d2_data), np.max(true_d2_data)]
    cmap_range6 = [np.min(f_data), np.max(f_data)]
    cmap_range7 = [np.min(true_f_data), np.max(true_f_data)]

    Mdisplay.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])
    #cbar.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])
    #cbar.draw_all()
    images_display.set_clim(vmin=cmap_range1[0], vmax=cmap_range1[1])
    #cbar1.set_clim(vmin=cmap_range1[0], vmax=cmap_range1[1])
    #cbar1.draw_all()
    d1_display.set_clim(vmin=cmap_range2[0], vmax=cmap_range2[1])
    true_d1_display.set_clim(vmin=cmap_range3[0], vmax=cmap_range3[1])

    d2_display.set_clim(vmin=cmap_range4[0], vmax=cmap_range4[1])
    true_d2_display.set_clim(vmin=cmap_range5[0], vmax=cmap_range5[1])

    f_display.set_clim(vmin=cmap_range6[0], vmax=cmap_range6[1])
    true_f_display.set_clim(vmin=cmap_range7[0], vmax=cmap_range7[1])
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')


    time.sleep(0.2)
    fig.canvas.draw()
    fig2.canvas.draw()

# Connect the key press event to the update function
fig.canvas.mpl_connect('key_press_event', on_key)



#matplotlib.use('TkAgg')





