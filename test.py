from utils import simulateDataset
import matplotlib.pyplot as plt

sim_data = simulateDataset(44)

d_1 = sim_data.d_1
d_2 = sim_data.d_2
sigma_g = sim_data.sigma_g

d_1 = d_1[0].squeeze(0)
d_2 = d_2[0].squeeze(0)
sigma_g = sigma_g[0].squeeze(0)

parameter_maps = [d_1, d_2, sigma_g]

fig, ax = plt.subplots(1, 3,  figsize=(18,5))
for i, para_map in enumerate(parameter_maps):
    ax[i].imshow(para_map, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
plt.show() 
