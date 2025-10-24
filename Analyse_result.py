from Analyse_result_compl import * #Comment this out if not available to you
from matplotlib import gridspec
import pickle
from pathlib import Path as Path
import matplotlib.ticker as mticker
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import time
import nibabel as nib
import re
from scipy.stats import ttest_ind
from matplotlib.path import Path as PPath  # NOT pathlib.Path

from IPython import embed
from scipy.stats import gaussian_kde


from math import sqrt
import torch
import pandas as pd
import nibabel as nib
from collections import defaultdict
from importlib import reload
from scipy.stats import gmean



plt_list = []


def to_normal_dict(d):
    if isinstance(d, defaultdict):  # Check if it's a defaultdict
        d = {k: to_normal_dict(v) for k, v in d.items()}  # Recursively convert
    return d
def nested_dict():
    return defaultdict(nested_dict)

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

def gamma(bval, theta, K):
    """
    torch gamma function
    """
    X = torch.float_power(1+theta*bval*1e-3,-K)+1e-6
    return X

def kurtosis(bval, D, K):
    """
    torch kurtosis function
    """

    X = torch.exp(-bval*D*1e-3+(bval*D*1e-3)**2*K/6+1e-6)

    return X


current_index = 0
vindex = 10

def index_deepest_folders(path, folder_list=None):
    """ Recursively index the deepest folders (folders without subfolders) inside a folder. """
    if folder_list is None:
        folder_list = []  # Initialize the list on first call
    path = Path(path)
    if path.is_dir():
        subfolders = [item for item in path.iterdir() if item.is_dir()]  # List of subdirectories
        if not subfolders:  # If no subdirectories, it's a deepest folder
            folder_list.append(str(path))
        else:
            for subfolder in subfolders:
                index_deepest_folders(subfolder, folder_list)  # Recurse into subdirectories
    return folder_list

def calc_mean(mean_array ,se_array,standard_error, pooled=False, no_std = False , normal_std = False):
    Mean = np.mean(mean_array)
    se_array = np.array(se_array) if not no_std else np.zeros(np.array(se_array).shape)
    #Mean = gmean(mean_array)

    M_num = len(mean_array)

    if standard_error:
        SE_i = np.array([n ** 2 for n in se_array])
        SE_Mean = np.sqrt(np.sum(SE_i / (M_num ** 2)))
    else:
        mu_ensemble = np.mean(mean_array)
        if pooled:
            SE_Mean = np.sqrt((np.sum(se_array**2)+ np.sum((mean_array-mu_ensemble)**2))/M_num)
        else:
            SE_Mean = np.sqrt((np.sum(se_array**2))/M_num)
    if no_std: SE_Mean = 0
    return Mean, SE_Mean if not normal_std else np.std(mean_array)

def in_docker():
    # Check for /.dockerenv (standard Docker flag)
    if os.path.exists('/.dockerenv'):
        return True
    # Check cgroup info for Docker or container runtime
    try:
        with open('/proc/1/cgroup', 'rt') as f: return any('docker' in line or 'kubepods' in line or 'containerd' in line for line in f)
    except Exception:
        return False

#Variables needed to define:

#MAIN_METHOD: Your main folder containing all your inference data: Main_folder/Neural-Network-Model/Diffusionmodel/Patient/run_x/
#FIGURE_SAVE_PATH: Define a path to save all you figures.
#run: Path(FIGURE_SAVE_PATH).mkdir(parents=True, exist_ok=True)
#patient: Define which patient
#patient_paths = {'patient': 'path',}#Define path to test data for each patient
#seg_paths = {'patient': 'path',}#Define path to you segmentation for ADC estimations for each patient
#true_paths = {'patient': 'path',} #path to obsidian result
#result_paths = {'patient': 'path'}  #path to inference data from you neural network models

if not local_computer:
    import scitool.dataview_devel.pyqtgraph_gui as plt2D
    reload(plt2D)
    matplotlib.use('Qt5Agg')


M_biexp,M_gamma,M_kurtosis,biexp_file, gamma_file, kurtosis_file = [None]*6
true_M_biexp,true_M_gamma,true_M_kurtosis = [None]*3
fig_comparison_plot,axes_comparison_plot = [None]*2
fig_adc_comparison_plot,axes_adc_comparison_plot, outer_gs = [None]*3
fig_res_comparison_plot,axes_res_comparison_plot = [None]*2
fig_method_comparison_plot,axes_method_comparison_plot = [None]*2
fig_dense_plot_ADC,axes_dense_plot_ADC = [None]*2

dict_runs_pixels_ADC = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: {"pixels": []})))))


table_vs_measured_loss_ADC = defaultdict(lambda:  defaultdict(lambda: defaultdict(lambda: {"clinical_loss": [],"true_loss": [], "measured_loss": []})))
table_ADC_print_toappend = defaultdict(lambda:  defaultdict(lambda: None))
table_ADC_print_toappend_flat = ''
table_ADC_print_dict = {
    "attention_unet": {
        "biexp": "A1",
        "kurtosis": "B1",
        "gamma": "C1"
    },
    "unet": {
        "biexp": "A2",
        "kurtosis": "B2",
        "gamma": "C2"
    },
    "res_atten_unet": {
        "biexp": "A3",
        "kurtosis": "B3",
        "gamma": "C3"
    }
}
table_ADC_print = f'''\\begin{{tabular}}{{|c|c|c|c|}}
    \\hline
     & Attention Unet & Unet & ResNET \\\\
    \\hline
    Biexponential & A1 & A2 & A3 \\\\
    \\hline
    Kurtosis & B1 & B2 & B3 \\\\
    \\hline
    Gamma & C1 & C2 & C3 \\\\
    \\hline
\\end{{tabular}}'''


def make_method_comparison(vindex = 12, new = True):#Comparison between noise-embed framework and main AI framework
    global fig_method_comparison_plot,axes_method_comparison_plot
    model, fitting, = 'res_atten_unet', 'kurtosis'
    includeDifference = True
    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]
    factor = 1000
    threshold = 0.005
    b100_index = [np.where(b_array == 100)[0][0]]
    b1000_index = [np.where(b_array == 1000)[0][0]]
    b100 = b_array[b100_index].numpy() / factor
    b1000 = b_array[b1000_index].numpy() / factor
    b_array = torch.tile(b_array, (3,))
    b = b_array.reshape(1, len(b_array), 1, 1)
    test_data_path = patient_paths[patient]
    seg_file_name = patient + '_seg.nii.gz'

    biexp_file = np.load(os.path.join(test_data_path, patient + '_biexp.npz'), allow_pickle=True)['arr_0'][()]

    b00 = torch.tensor(biexp_file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)
    test_seg_path = seg_paths[patient] + seg_file_name
    roi_file = nib.load(os.path.join(test_seg_path))
    roi_file = np.asanyarray(roi_file.dataobj).transpose(2, 1, 0)
    region_mask = roi_file == 4
    n_slices = b00.shape[0]
    b0_mean = np.mean((b00[:, 0][region_mask[:n_slices]]).numpy())
    b00 = b00[:, 0, 20:-20, :]
    print(f'Thresold pixel value: {b0_mean * threshold}')

    n_rows = 2 if not includeDifference else 3
    n_cols = 5 if fitting != 'biexp' else 5

    # Create figure
    fig_method_comparison_plot = plt.figure(figsize=(8.4, 2.76 if not includeDifference else 4))
    gs = gridspec.GridSpec(n_rows, n_cols + 2, width_ratios=[1] * n_cols + [0.02] + [0.07])
    fig_method_comparison_plot.canvas.manager.set_window_title(f'{model}-{fitting}-slice {vindex}')

    axes_method_comparison_plot = []
    for i in range(n_rows):
        row_axes = []
        for j in range(n_cols):
            ax = fig_method_comparison_plot.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes_method_comparison_plot.append(row_axes)
    axes_method_comparison_plot = np.array(axes_method_comparison_plot)
    cbar_ax = fig_method_comparison_plot.add_subplot(gs[-1, -1])  # last column for colorbar


    methods = ['cross_validation_l1_s0est_3D','cross_validation_l1_s0est_3D_feed_sigma']

    method_names = [f'Main AI framework\n(Without noise input)','Noise-embed\nframework']

    bvalues = [1000.,2000.]

    bs = np.tile(np.linspace(100, 2000, 20),3)

    DifferenceDict = {}
    for mi,m in enumerate(methods):

        pars_arr = np.zeros(shape=(5,4,22,200,240))
        Ms_arr = np.zeros(shape=(5,2,200,240))
        for ri, run in enumerate(['run_1','run_2','run_3','run_4','run_5']):
            p = Path(os.path.join(result_paths, m, model, fitting, patient, run))
            print(p)

            pars = np.load(os.path.join(p,'parameters.npy'))
            if '3D' in m:
                adcks = np.mean(pars[[0,2,4]], axis = 0)
                ks =np.mean(pars[[1,3,5]], axis = 0)

                pars = np.array([adcks,ks,pars[-2], pars[-1]])
            else:
                pars = pars[:]

            M = np.load(os.path.join(p,'M.npy'))

            if not '3D' in m:
                n_slices = M.shape[0]//3
                M = np.concatenate((M[0:n_slices], M[n_slices:n_slices * 2], M[n_slices * 2:n_slices * 3]), axis=1)
            else:
                n_slices = M.shape[0]

            Ms = np.empty(shape=(len(bvalues),*M.shape[-2:]))
            print(f'ms shape {Ms.shape}')
            for bi,b in enumerate(bvalues):
                indices = np.where(np.isin(bs,[b]))[0]
                print(indices)
                print(M.shape)

                Mtemp = np.mean(M[vindex,indices],axis = 0)
                print(M.shape)
                Ms[bi] = Mtemp/Mtemp.max()

            pars_arr[ri] = pars
            Ms_arr[ri] = Ms

        pars = np.mean(pars_arr,axis = 0)
        Ms = np.mean(Ms_arr,axis = 0)

        pars[0][b00 < b0_mean * threshold] = 0
        pars[1][b00 < b0_mean * threshold] = 0
        pars[2][b00 < b0_mean * threshold] = 0
        pars[3][b00 < b0_mean * threshold] = 0
        Ms[0][b00[vindex] < b0_mean * threshold] = 0
        Ms[1][b00[vindex] < b0_mean * threshold] = 0

        pars[-2] = pars[-2]/pars[-2].max()

        DifferenceDict[m] = (Ms,pars)

    for mi, m in enumerate(methods):

        Ms, pars = DifferenceDict[m]
        if includeDifference:
            Ms1, pars1 = DifferenceDict[methods[0]]
            Ms2, pars2 = DifferenceDict[methods[1]]
            MsDiff, parsDiff = (Ms1-Ms2)/Ms1, (pars1-pars2)/pars1
            MsDiff, parsDiff = np.where(Ms1 != 0, MsDiff, 0)*100,np.where(pars1 != 0, parsDiff, 0)*100

            vmaxDiff , vminDiff = 30, -30
        for i,name in enumerate(method_names):

            axes_method_comparison_plot[i,0].set_ylabel(method_names[i], fontsize = 9, rotation = 90, labelpad = 10)

        if includeDifference:
            axes_method_comparison_plot[-1, 0].set_ylabel('Difference maps', fontsize=9, rotation=90, labelpad=10)

        axes_method_comparison_plot[0,0].set_title('b₀')
        axes_method_comparison_plot[0,1].set_title('b₁₀₀₀')
        axes_method_comparison_plot[0,2].set_title('b₂₀₀₀')

        if fitting == 'biexp':
            axes_method_comparison_plot[0,len(bvalues)+1].set_title('D₁')
            axes_method_comparison_plot[0,len(bvalues)+2].set_title('D₂')
            axes_method_comparison_plot[0,len(bvalues)+3].set_title('f')
            axes_method_comparison_plot[mi, len(bvalues)+1].imshow(pars[0,vindex], cmap='gray',vmax=4, vmin=0)
            axes_method_comparison_plot[mi, len(bvalues)+2].imshow(pars[1,vindex], cmap='gray',vmax=1, vmin=0)
            axes_method_comparison_plot[mi, len(bvalues)+3].imshow(pars[2,vindex], cmap='gray',vmax=1, vmin=0)
    
        elif fitting == 'kurtosis':
            axes_method_comparison_plot[0, len(bvalues)+1].set_title(r'$\mathrm{ADC}_{\mathrm{K}}$')
            axes_method_comparison_plot[0, len(bvalues)+2].set_title('K')
            axes_method_comparison_plot[mi, len(bvalues)+1].imshow(pars[0,vindex], cmap='gray',vmax=4, vmin=0)
            axes_method_comparison_plot[mi, len(bvalues)+2].imshow(pars[1,vindex], cmap='gray',vmax=3, vmin=0)
            if includeDifference:
                axes_method_comparison_plot[-1, len(bvalues) + 1].imshow(parsDiff[0, vindex], cmap='viridis', vmax=vmaxDiff,vmin=vminDiff)
                axes_method_comparison_plot[-1, len(bvalues) + 2].imshow(parsDiff[1, vindex], cmap='viridis', vmax=vmaxDiff,vmin=vminDiff)

            #Write for gamma if you want

            
        else: 
            assert False, f'fitting name {fitting} is invalid'

        axes_method_comparison_plot[mi, 0].imshow(pars[-2,vindex], cmap='gray', vmax = 0.25)
        if includeDifference:
            axes_method_comparison_plot[-1, 0].imshow(parsDiff[-2, vindex], cmap='viridis',vmax=vmaxDiff,vmin=vminDiff)
        for bi,b in enumerate(bvalues):

            axes_method_comparison_plot[mi, bi+1].imshow(Ms[bi],vmax = 0.65, cmap='gray')
            if includeDifference:
                im = axes_method_comparison_plot[-1, bi + 1].imshow(MsDiff[bi], vmax=vmaxDiff,vmin=vminDiff,cmap='viridis')


        for ax in axes_method_comparison_plot[mi, :].flatten():
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])

        if includeDifference:
            for ax in axes_method_comparison_plot[-1, :].flatten():
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])

    cbar = fig_method_comparison_plot.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([vminDiff,0,vmaxDiff ])
    cbar.set_ticklabels([f'{vminDiff}', '', f'{vmaxDiff}'])  # set labels

    cbar.set_label(r'$\Delta_{\mathrm{rel}}$ [%]', fontsize=9, labelpad=-2)
    cbar.ax.tick_params(labelsize=7)  # font size for tick labels
    cbar.ax.yaxis.label.set_rotation(270)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0)
    plt.subplots_adjust(wspace=0.05, hspace=0.02)  # Reduce white space between images
    path_to_save_png = os.path.join(FIGURE_SAVE_PATH, 'feed_sigma_comparison.png')
    print('Saved')
    path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, 'feed_sigma_comparison.pdf')
    plt.savefig(path_to_save_png, bbox_inches='tight')
    plt.savefig(path_to_save_pdf, bbox_inches='tight', dpi = 200)

def make_noisemap_comparison(run, vindex=12, new=True):#Comparison between OBSIDIAN estimated sigma and estimation from DNN model (guided)
    global fig_method_comparison_plot, axes_method_comparison_plot
    model, fitting, = 'res_atten_unet', 'kurtosis'

    if fig_method_comparison_plot is None or new:
        fig_method_comparison_plot, axes_method_comparison_plot = plt.subplots(3, 4,
                                                                               figsize=(8.4, 5.2))  # 7.95
        fig_method_comparison_plot.canvas.manager.set_window_title(f'{model}-{fitting}-run{run} - slice {vindex}')
    if run is not None:
        runs = ['run_' + str(run)]
    else:
        runs = [f'run_{s+1}' for s in range(5)]

    methods = ['cross_validation_l1_s0est_3D_feed_and_est_sigma','cross_validation_l1_s0est_3D_est_sigma' ]

    method_names = ['OBSIDIAN', f'Non-blinded', f'Blinded']

    bvalues = [1000., 2000.]

    bs = np.tile(np.linspace(100, 2000, 20), 3)

    for mi, m in enumerate(methods):
        predsigma_array = []
        Ms_array = []
        for run in runs:
            p = Path(os.path.join(result_paths, m, model, fitting, patient, run))
            print(p)

            if 'parameters.npy' in os.listdir(p):
                pars = np.load(os.path.join(p, 'parameters.npy'))
                pars = pars[-2]

                pars = pars / pars.max()
                predsigma = np.load(os.path.join(p, 'estimated_sigma.npy'))
                M = np.load(os.path.join(p, 'M.npy'))

                if not '3D' in m:
                    n_slices = M.shape[0] // 3
                    M = np.concatenate((M[0:n_slices], M[n_slices:n_slices * 2], M[n_slices * 2:n_slices * 3]), axis=1)
                else:
                    n_slices = M.shape[0]

                Ms = np.empty(shape=(len(bvalues), *M.shape[-2:]))
                for bi, b in enumerate(bvalues):
                    indices = np.where(np.isin(bs, [b]))[0]

                    Mtemp = np.mean(M[vindex, indices], axis=0)
                    Ms[bi] = Mtemp / Mtemp.max()
            else:
                assert False, f'Did not find parameters.npy in {p}'

            predsigma_array.append(predsigma)
            Ms_array.append(Ms)

        Ms = np.mean(np.array(Ms_array), axis = 0)
        predsigma= np.mean(np.array(predsigma_array), axis = 0)

        for i, name in enumerate(method_names):
            axes_method_comparison_plot[i, 0].set_ylabel(method_names[i], fontsize=9, rotation=90, labelpad=10)

        axes_method_comparison_plot[0, 0].set_title('b₀')
        axes_method_comparison_plot[0, 1].set_title('b₁₀₀₀')
        axes_method_comparison_plot[0, 2].set_title('b₂₀₀₀')


        if fitting == 'kurtosis':
            axes_method_comparison_plot[0, len(bvalues) + 1].set_title('Noise map')

            axes_method_comparison_plot[mi + 1, len(bvalues) + 1].imshow(predsigma[vindex,0], cmap='gray', vmax=30,
                                                                         vmin=0)  
        else:
            assert False, f'fitting name {fitting} is invalid'

        axes_method_comparison_plot[mi + 1, 0].imshow(pars[vindex], cmap='gray',
                                                      vmax=0.25)  
        for bi, b in enumerate(bvalues):
            axes_method_comparison_plot[mi + 1, bi + 1].imshow(Ms[bi], vmax=0.65,
                                                               cmap='gray')  

        for ax in axes_method_comparison_plot[mi + 1, :].flatten():
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])

    obs_names = ['OBSIDIAN']  # ['OBSIDIAN (No bias correction)', 'OBSIDIAN']
    obs_file_name = ['']  # ['_nb', '']
    obs_result_name = ['3Dsig']  # ['3D', '3Dsig']

    num_diff = 3
    test_data_path = patient_paths[patient]
    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]
    b = b_array.reshape(1, len(b_array), 1, 1)
    var_list = {}
    for idx, name in enumerate(obs_names):
        obs_collect = torch.zeros(size=(num_diff, n_slices, 20, 240, 240))
        obs_collectb0 = torch.zeros(size=(num_diff, n_slices, 1, 240, 240))
        for i in range(num_diff):

            result_file = np.load(os.path.join(test_data_path, patient + '_' + fitting + obs_file_name[idx] + '.npz'),
                                  allow_pickle=True)['arr_0'][()]

            if fitting == 'kurtosis':
                true_D = torch.from_numpy(
                    result_file['result'][obs_result_name[idx]][:, :, :, 2 * i + 0])  ###Maybe expand to other direction
                true_K = torch.from_numpy(result_file['result'][obs_result_name[idx]][:, :, :, 2 * i + 1])
                obssigma = torch.tensor(result_file['result'][obs_result_name[idx]][:, :, :, -2],
                                     dtype=torch.float32)
                if i == 1:
                    var_list[f'sigma'] = obssigma[vindex, 20:-20].numpy()
                obsb0 = torch.tensor(result_file['result'][obs_result_name[idx]][:, :, :, -3],
                                     dtype=torch.float32).unsqueeze(dim=1)
                true_M = obsb0 * kurtosis(D=true_D.unsqueeze(dim=1), K=true_K.unsqueeze(dim=1), bval=b)
                obs_collect[i] = true_M
                obs_collectb0[i] = obsb0

        true_M_avg = np.mean(obs_collect.numpy(), axis=0)
        true_b0_avg = np.mean(obs_collectb0.numpy(), axis=0)
        obsb1000 = true_M_avg[vindex, 9, 20:-20, :]
        obsb2000 = true_M_avg[vindex, 19, 20:-20, :]
        obsb0 = true_b0_avg[vindex, 0, 20:-20]

    axes_method_comparison_plot[0, 0].imshow(obsb0 / obsb0.max(), cmap='gray', vmax=0.25)  
    axes_method_comparison_plot[0, 1].imshow(obsb1000 / obsb1000.max(), vmax=0.65,
                                             cmap='gray')  
    axes_method_comparison_plot[0, 2].imshow(obsb2000 / obsb2000.max(), vmax=0.65,
                                             cmap='gray')  
    for i, (k, var) in enumerate(var_list.items()):
        vmax = var.max()
        axes_method_comparison_plot[0, i + 3].imshow(var, cmap='gray', vmax=30, vmin=0)

    for i in range(4):
        axes_method_comparison_plot[0, i].get_yaxis().set_ticks([])
        axes_method_comparison_plot[0, i].get_xaxis().set_ticks([])

    plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce white space between images
    path_to_save_png = os.path.join(FIGURE_SAVE_PATH, 'sigma_comparison.png')
    path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, 'sigma_comparison.pdf')
    plt.savefig(path_to_save_png, bbox_inches='tight')
    plt.savefig(path_to_save_pdf, bbox_inches='tight')

def comparison_figure(patient,bvalues, vindex, fitting_model,new = False):#Comparison in DWI at different b values
    global biexp_file, gamma_file, kurtosis_file
    global fig_comparison_plot,axes_comparison_plot
    global true_M_biexp,true_M_gamma,true_M_kurtosis
    print(f'Loading {fitting_model} ...')

    test_data_path = patient_paths[patient]

    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False
    if fig_comparison_plot is None or new:
        fig_comparison_plot,axes_comparison_plot = plt.subplots(len(bvalues), 8, figsize=(14.3, 7.66))
        print(f'Making the plot...')
        fig_comparison_plot.canvas.manager.set_window_title(f'{fitting_model} - slice {vindex}')

    biexp_file = np.load(os.path.join(test_data_path, patient+'.npy'), allow_pickle=True)[()]
    n_slices = biexp_file['image_b0'].shape[0]

    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]

    indices = [np.where(b_array == target_value)[0][0] for target_value in bvalues]
    indices_clinical = [np.where(b_array == target_value)[0][0] for target_value in [100.,1000.,1500.]]

    b = b_array.reshape(1, len(b_array), 1, 1)

    b0 = torch.tensor(biexp_file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)
    shape = (1, b0.shape[2] - 40, b0.shape[3])


    clinical_file = np.load(os.path.join(test_data_path, patient + '_clinical.npz'), allow_pickle=True)
    clinical_file = clinical_file['arr_0'][()]
    M_clinical = clinical_file['image_data'][:, :, 20:-20, :]



    if M_clinical.shape[1]>22:
        print('Warning clinical data has more than 22 slices')
    n_slices = min(n_slices,M_clinical.shape[1])
    obsidian_dict = {'OBSIDIAN': None, 'OBSIDIAN (No bias correction)': None}
    obs_names = ['OBSIDIAN (No bias correction)','OBSIDIAN']
    obs_file_name = ['_nb','']
    obs_result_name = ['3D','3Dsig']

    num_diff = 3
    for idx,name in enumerate(obs_names):
        if fitting_model == 'biexp':biexp_file=None
        if fitting_model == 'kurtosis':kurtosis_file=None
        if fitting_model == 'gamma':gamma_file=None

        obs_array = np.empty(shape=(num_diff,5, *shape[1:]))
        obs_array_save = np.empty(shape=(num_diff,n_slices,20, *shape[1:]))

        for i in range(num_diff):
            print(f'{i}')


            if fitting_model == 'biexp':
                if biexp_file is None: biexp_file = np.load(os.path.join(test_data_path, patient + '_biexp' + obs_file_name[idx] + '.npz'),
                                     allow_pickle=True)['arr_0'][()]
                true_d1 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+0])  ###Maybe expand to other direction
                true_d2 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+1])
                true_f = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+2])
                b0 = torch.tensor(biexp_file['result'][obs_result_name[idx]][:, :, :,-3], dtype=torch.float32).unsqueeze(dim=1)
                true_M_biexp = b0 * bio_exp(d1=true_d1.unsqueeze(dim=1), d2=true_d2.unsqueeze(dim=1),
                                            f=true_f.unsqueeze(dim=1),
                                            b=b)
                obs_array_save[i] = true_M_biexp[:,:,20:-20,:].numpy()
                obs_array[i] = true_M_biexp[vindex,:,20:-20,:][indices]
            if fitting_model == 'kurtosis':
                if kurtosis_file is None:
                    kurtosis_file = np.load(os.path.join(test_data_path, patient + '_kurtosis' + obs_file_name[idx] + '.npz'),
                            allow_pickle=True)['arr_0'][()]
                true_d = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                true_k = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                b0 = torch.tensor(kurtosis_file['result'][obs_result_name[idx]][:, :, :, -3],
                                  dtype=torch.float32).unsqueeze(dim=1)
                true_M_kurtosis = b0 * kurtosis(bval=b, D=true_d.unsqueeze(dim=1), K=true_k.unsqueeze(dim=1))
                obs_array_save[i] = true_M_kurtosis[:, :, 20:-20, :]
                obs_array[i] = true_M_kurtosis[vindex, :, 20:-20, :][indices]
            if fitting_model == 'gamma':
                if gamma_file is None:
                    gamma_file = np.load(os.path.join(test_data_path, patient + '_gamma' + obs_file_name[idx] + '.npz'),
                                         allow_pickle=True)['arr_0'][()]
                true_k = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                true_theta = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                b0 = torch.tensor(gamma_file['result'][obs_result_name[idx]][:, :, :, -3],
                                  dtype=torch.float32).unsqueeze(dim=1)
                true_M_gamma = b0 * gamma(bval=b, K=true_k.unsqueeze(dim=1), theta=true_theta.unsqueeze(dim=1))
                obs_array_save[i] = true_M_gamma[:, :, 20:-20, :]
                obs_array[i] =true_M_gamma[vindex, :, 20:-20, :][indices]

        obs_img =  gmean(obs_array, axis = 0)
        obsidian_dict[obs_names[idx]] =  obs_img

    biexp_file = np.load(os.path.join(test_data_path, patient+'_biexp'+'.npz'), allow_pickle=True)['arr_0'][()]

    real_array = np.empty(shape=(num_diff,5, *shape[1:]))
    for i in range(num_diff):
        real_array[i] = biexp_file['image']['3Dsig'][vindex,:,20:-20,:][list(20*i+np.array(indices))]
    real_M = gmean(real_array,axis=0)

    vmax = [0.55 * m.max() for m in real_M]  # 0.8 innan
    vmax[0] = 0.50 * real_M[0].max()  # 0.45
    vmax[1] = 0.7 * real_M[1].max()  # 0.6

    vmin = [m.min() for m in real_M]
    models = [MAIN_METHOD+'/'+m for m in ['unet','res_atten_unet','attention_unet']] + [MAIN_METHOD+'_no_rice/unet']#, 'transformer']
    model_names_list = ['unet','res_atten_unet','attention_unet', 'unet_rician']
    predicted_images = {}
    for i, model in enumerate(models):
        model_name = model_names_list[i]
        result_path_base = result_paths+model+'/'+fitting_model+'/'+patient
        run_array = np.empty(shape=(len(os.listdir(result_path_base)),5, *shape[1:]))
        run_array_save = np.empty(shape=(len(os.listdir(result_path_base)),n_slices,20, *shape[1:]))

        for ri, run, in enumerate(os.listdir(result_path_base)):
            result_path = os.path.join(result_path_base, run)
            M = np.load(f'{result_path}/M.npy',allow_pickle=True)

            if not '3D' in MAIN_METHOD: M_con = np.concatenate((M[0:n_slices], M[n_slices:n_slices*2], M[n_slices*2:n_slices*3]), axis=1)
            else: M_con = M
            first_20 = M_con[:, :20, :, :]  # First 20 images
            second_20 = M_con[:, 20:40, :, :]  # Second 20 images
            third_20 = M_con[:, 40:60, :, :]  # Third 20 images

            run_array_save[ri] = gmean(np.array([first_20,second_20,third_20]),axis =0)
            diff_array = np.empty(shape=(3, 5, *shape[1:]))
            for di in range(3):
                diff_array[di] = M_con[vindex, 20*di+0:20*di+20, :, :][indices]
            run_array[ri] = gmean(diff_array, axis=0)
        pred_img= gmean(run_array, axis = 0)
        predicted_images[model_name] =pred_img



    fontsize = 12
    for ax in axes_comparison_plot.flatten():
        ax.axis('off')  # Turns off x-axis, y-axis, and ticks
    axes_comparison_plot[0, 0].set_title(f'Clinical Image', fontsize = fontsize)
    axes_comparison_plot[0, 1].set_title(f'Raw Image', fontsize = fontsize)
    axes_comparison_plot[0, 2].set_title(f'Direct fit', fontsize=fontsize)
    axes_comparison_plot[0, 3].set_title(f'Standard U-Net\n(Direct)', fontsize = fontsize)
    axes_comparison_plot[0, 4].set_title(f'OBSIDIAN', fontsize = fontsize)
    axes_comparison_plot[0, 5].set_title(f'Standard U-Net', fontsize = fontsize)
    axes_comparison_plot[0, 6].set_title(f'Attention U-Net', fontsize = fontsize)
    axes_comparison_plot[0, 7].set_title(f'Residual Attention U-Net', fontsize=fontsize)


    for i,index in enumerate(indices):
        axes_comparison_plot[i, 0].axis('on')  # Turns off x-axis, y-axis, and ticks
        axes_comparison_plot[i, 0].get_xaxis().set_ticks([])
        axes_comparison_plot[i, 0].get_yaxis().set_ticks([])
        axes_comparison_plot[i, 0].set_ylabel(f'{int(b_array[index])}', fontsize = fontsize)
    index_for_clinical={'0':1,
                       '2':2,
                       '3':3}

    for ax in axes_comparison_plot.flat:
        ax.tick_params(direction='out', length=3, pad=2)
        ax.margins(0)
        ax.set_xlim(0, 240)
        ax.set_ylim(200, 0)


    for i,obsidian_img in enumerate(obsidian_dict['OBSIDIAN']):
        plot_idx=i

        if indices[i] in indices_clinical:
            cshow = M_clinical[index_for_clinical[str(i)]][vindex]
            cvmax = [0.55*cshow.max(),0.6,0.75*cshow.max(),0.7*cshow.max(),0.6]
            axes_comparison_plot[plot_idx, 0].imshow(cshow, cmap='gray', vmax=0.8*cvmax[i],vmin=0)  
        else:
            axes_comparison_plot[plot_idx, 0].imshow(np.ones(shape=(300,300)), cmap='gray', vmax=1,vmin=0.99)  


        axes_comparison_plot[plot_idx, 1].imshow(real_M[i], cmap='gray', vmax=vmax[i],vmin=vmin[i]) 

        axes_comparison_plot[plot_idx, 2].imshow(obsidian_dict['OBSIDIAN (No bias correction)'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])  
        axes_comparison_plot[plot_idx, 3].imshow(predicted_images['unet_rician'][i], cmap='gray', vmax=vmax[i],
                                                 vmin=vmin[i])
        axes_comparison_plot[plot_idx, 4].imshow(obsidian_img, cmap='gray', vmax=vmax[i],vmin=vmin[i])  

        axes_comparison_plot[plot_idx, 5].imshow(predicted_images['unet'][i],cmap='gray', vmax=vmax[i],vmin=vmin[i])

        axes_comparison_plot[plot_idx, 6].imshow(predicted_images['attention_unet'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])  

        axes_comparison_plot[plot_idx, 7].imshow(predicted_images['res_atten_unet'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0)
    plt.subplots_adjust(wspace=0.01, hspace=0.02)  # Reduce white space between images

def make_comparison_figure():
    global fig_comparison_plot, patient
    bvalues = [100.,500.,1000.,1500.,2000.]
    for fit in ['biexp','kurtosis', 'gamma']:
       print(f'Making figure for {fit}')
       comparison_figure(patient,bvalues, vindex=12, fitting_model=fit, new=True)
       path_to_save_png = os.path.join(FIGURE_SAVE_PATH, f'{fit}.png')
       path_to_save_eps = os.path.join(FIGURE_SAVE_PATH, f'{fit}.eps')
       plt.savefig(path_to_save_png, bbox_inches='tight')
       plt.savefig(path_to_save_eps, format='eps', bbox_inches='tight')

def comparison_figure_reduced(patient , bvalues, vindex, fitting_model,new = False, dual_patients = False, second_pat = False, vmaxc=None):
    global biexp_file, gamma_file, kurtosis_file
    global fig_comparison_plot,axes_comparison_plot
    global true_M_biexp,true_M_gamma,true_M_kurtosis
    print(f'Loading {fitting_model} {patient}...')
    test_data_path = patient_paths[patient]

    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False
    if fig_comparison_plot is None or new:
        fig_comparison_plot = plt.figure(figsize=(11.23, 1.255737704918033*(2*len(bvalues)+0.1)))
        fig_comparison_plot.canvas.manager.set_window_title(f'{fitting_model} - slice {vindex}')

        if 1==1:

            gs = gridspec.GridSpec(len(bvalues) if not dual_patients else len(bvalues)*2+1, 9, figure=fig_comparison_plot,
                                   height_ratios=[1, 1,1, 1, 0.1,1, 1, 1, 1],
                                   width_ratios = [1,1,1,1,0.05,1,1,1,1])

            axes_comparison_plot = []
            for i in range(len(bvalues)):
                row_axes = []
                for ji,j in enumerate([0,1,2,3,5,6,7,8]):
                    ax = fig_comparison_plot.add_subplot(gs[i, j])
                    row_axes.append(ax)
                axes_comparison_plot.append(row_axes)

            ax_spacer = fig_comparison_plot.add_subplot(gs[4, :])
            ax_spacer.axhline(0.5, color='black', linewidth=1)
            ax_spacer.set_xticks([])
            ax_spacer.set_yticks([])
            ax_spacer.spines['top'].set_visible(False)
            ax_spacer.spines['right'].set_visible(False)
            ax_spacer.spines['left'].set_visible(False)
            ax_spacer.spines['bottom'].set_visible(False)

            for i in range(len(bvalues)+1, len(bvalues)*2+1):
                row_axes = []

                for ji, j in enumerate([0, 1, 2, 3, 5, 6, 7, 8]):
                    ax = fig_comparison_plot.add_subplot(gs[i, j])
                    row_axes.append(ax)
                axes_comparison_plot.append(row_axes)

    biexp_file = np.load(os.path.join(test_data_path, patient+'.npy'), allow_pickle=True)[()]

    n_slices = biexp_file['image_b0'].shape[0]
    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]

    indices = [np.where(b_array == target_value)[0][0] for target_value in bvalues]
    indices_clinical = [np.where(b_array == target_value)[0][0] for target_value in [100.,1000.,1500.]]

    b = b_array.reshape(1, len(b_array), 1, 1)

    b0 = torch.tensor(biexp_file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)
    shape = (1, b0.shape[2] - 40, b0.shape[3])


    clinical_file = np.load(os.path.join(test_data_path, patient + '_clinical.npz'), allow_pickle=True)
    clinical_file = clinical_file['arr_0'][()]
    M_clinical = clinical_file['image_data'][:, :, 20:-20, :]


    n_slices = min(n_slices, M_clinical.shape[1])

    obsidian_dict = {'OBSIDIAN': None, 'OBSIDIAN (No bias correction)': None}
    obs_names = ['OBSIDIAN (No bias correction)','OBSIDIAN']
    obs_file_name = ['_nb','']
    obs_result_name = ['3D','3Dsig']

    num_diff = 3
    for idx,name in enumerate(obs_names):

        if fitting_model == 'biexp':biexp_file=None
        if fitting_model == 'kurtosis':kurtosis_file=None
        if fitting_model == 'gamma':gamma_file=None

        obs_array = np.empty(shape=(num_diff,len(bvalues), *shape[1:]))
        print(n_slices)
        obs_array_save = np.empty(shape=(num_diff,n_slices,20, *shape[1:]))
        print(obs_array_save.shape)
        for i in range(num_diff):
            print(f'{i}')

            if fitting_model == 'biexp':
                if biexp_file is None: biexp_file = np.load(os.path.join(test_data_path, patient + '_biexp' + obs_file_name[idx] + '.npz'),
                                     allow_pickle=True)['arr_0'][()]
                true_d1 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+0])  ###Maybe expand to other direction
                true_d2 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+1])
                true_f = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+2])
                b0 = torch.tensor(biexp_file['result'][obs_result_name[idx]][:, :, :,-3], dtype=torch.float32).unsqueeze(dim=1)
                true_M_biexp = b0 * bio_exp(d1=true_d1.unsqueeze(dim=1), d2=true_d2.unsqueeze(dim=1),
                                            f=true_f.unsqueeze(dim=1),
                                            b=b)
                obs_array_save[i] = true_M_biexp[:,:,20:-20,:].numpy()
                obs_array[i] = true_M_biexp[vindex,:,20:-20,:][indices]
            if fitting_model == 'kurtosis':
                if kurtosis_file is None:
                    kurtosis_file = np.load(os.path.join(test_data_path, patient + '_kurtosis' + obs_file_name[idx] + '.npz'),
                            allow_pickle=True)['arr_0'][()]
                true_d = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                true_k = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                b0 = torch.tensor(kurtosis_file['result'][obs_result_name[idx]][:, :, :, -3],
                                  dtype=torch.float32).unsqueeze(dim=1)
                true_M_kurtosis = b0 * kurtosis(bval=b, D=true_d.unsqueeze(dim=1), K=true_k.unsqueeze(dim=1))
                obs_array_save[i] = true_M_kurtosis[:, :, 20:-20, :]
                obs_array[i] = true_M_kurtosis[vindex, :, 20:-20, :][indices]
            if fitting_model == 'gamma':
                if gamma_file is None:
                    gamma_file = np.load(os.path.join(test_data_path, patient + '_gamma' + obs_file_name[idx] + '.npz'),
                                         allow_pickle=True)['arr_0'][()]
                true_k = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                true_theta = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                b0 = torch.tensor(gamma_file['result'][obs_result_name[idx]][:, :, :, -3],
                                  dtype=torch.float32).unsqueeze(dim=1)
                true_M_gamma = b0 * gamma(bval=b, K=true_k.unsqueeze(dim=1), theta=true_theta.unsqueeze(dim=1))
                obs_array_save[i] = true_M_gamma[:, :, 20:-20, :]
                obs_array[i] =true_M_gamma[vindex, :, 20:-20, :][indices]

        obsidian_dict[obs_names[idx]] =  gmean(obs_array, axis = 0)



    biexp_file = np.load(os.path.join(test_data_path, patient+'_biexp'+'.npz'), allow_pickle=True)['arr_0'][()]

    real_array = np.empty(shape=(num_diff,len(bvalues), *shape[1:]))
    for i in range(num_diff):
        real_array[i] = biexp_file['image']['3Dsig'][vindex,:,20:-20,:][list(20*i+np.array(indices))]
    real_M = gmean(real_array,axis=0)

    def middle_means(images):
        roi_height = 20
        roi_width = 20

        # Compute starting indices for the central crop
        center_y = 200 // 2
        center_x = 240 // 2
        start_y = center_y - roi_height // 2
        start_x = center_x - roi_width // 2

        # Extract the ROI for each image
        roi = images[:, start_y:start_y + roi_height, start_x:start_x + roi_width]

        # Compute the mean within the ROI for each image
        return roi.mean(axis=(1, 2))

    roi_means = middle_means(real_M)
    vmax = [1.5 * m.max() for m in roi_means]  # 0.8 innan
    vmin = [m.min() for m in real_M]


    models = [MAIN_METHOD+'/'+ word for word in ['attention_unet','unet','res_atten_unet']] + [MAIN_METHOD+'_no_rice/unet']
    model_name_list = ['attention_unet','unet','res_atten_unet','unet_rician']

    predicted_images = {}
    for i, model in enumerate(models):
        model_name = model_name_list[i]
        result_path_base = result_paths+model+'/'+fitting_model+'/'+patient
        run_array = np.empty(shape=(len(os.listdir(result_path_base)),len(bvalues), *shape[1:]))
        run_array_save = np.empty(shape=(len(os.listdir(result_path_base)),n_slices,20, *shape[1:]))

        for ri, run, in enumerate(os.listdir(result_path_base)):
            result_path = os.path.join(result_path_base, run)
            M = np.load(f'{result_path}/M.npy',allow_pickle=True)

            if not '3D' in MAIN_METHOD: M_con = np.concatenate((M[0:n_slices], M[n_slices:n_slices*2], M[n_slices*2:n_slices*3]), axis=1)
            else: M_con = M
            first_20 = M_con[:, :20, :, :]  # First 20 images
            second_20 = M_con[:, 20:40, :, :]  # Second 20 images
            third_20 = M_con[:, 40:60, :, :]  # Third 20 images

            run_array_save[ri] = gmean(np.array([first_20,second_20,third_20]),axis =0)
            diff_array = np.empty(shape=(3, len(bvalues), *shape[1:]))
            for di in range(3):
                diff_array[di] = M_con[vindex, 20*di+0:20*di+20, :, :][indices]
            run_array[ri] = gmean(diff_array, axis=0)
        predicted_images[model_name] = gmean(run_array, axis = 0)


    fontsize = 13
    for ax in np.array(axes_comparison_plot).flatten():
        ax.axis('off')  # Turns off x-axis, y-axis, and ticks
    for ax in np.array(axes_comparison_plot)[:,0].flatten():
        ax.axis('on')  # Turns off x-axis, y-axis, and ticks

    axes_comparison_plot[0][0].set_title(f'Clinical Image', fontsize = fontsize)
    axes_comparison_plot[0][1].set_title(f'Raw Image', fontsize = fontsize)
    axes_comparison_plot[0][2].set_title(f'Direct fit', fontsize=fontsize)
    axes_comparison_plot[0][3].set_title(f'Standard U-Net\n(Direct)', fontsize = fontsize)
    axes_comparison_plot[0][4].set_title(f'OBSIDIAN', fontsize = fontsize)
    axes_comparison_plot[0][5].set_title(f'Standard U-Net', fontsize = fontsize)
    axes_comparison_plot[0][6].set_title(f'Attention\nU-Net', fontsize = fontsize)
    axes_comparison_plot[0][7].set_title(f'Residual\nAttention\nU-Net', fontsize=fontsize)


    for i,index in enumerate(indices):
        plot_idx = i
        if dual_patients and second_pat:
            plot_idx = i+len(bvalues)

        axes_comparison_plot[plot_idx][0].axis('on')  # Turns off x-axis, y-axis, and ticks
        axes_comparison_plot[plot_idx][0].get_xaxis().set_ticks([])
        axes_comparison_plot[plot_idx][0].get_yaxis().set_ticks([])
        axes_comparison_plot[plot_idx][0].set_ylabel(f'{int(b_array[index])}', fontsize = fontsize)
    index_for_clinical={'0':1,#
                        '1':2,# 3
                       '2':3}#del

    for ax in np.array(axes_comparison_plot).flatten():
        ax.tick_params(direction='out', length=3, pad=2)
        ax.margins(0)
        ax.set_xlim(0, 240)
        ax.set_ylim(200, 0)


    c_means =  middle_means(M_clinical[:,vindex])

    for i,obsidian_img in enumerate(obsidian_dict['OBSIDIAN']):
        plot_idx=i

        if dual_patients and second_pat:
            plot_idx = plot_idx + len(bvalues)

        if indices[i] in indices_clinical:

            c_show = M_clinical[index_for_clinical[str(i)]][vindex]
            cvmax = vmaxc[1] * np.array(c_means)

            cvmax[0] = vmaxc[0] * c_means[0]
            axes_comparison_plot[plot_idx][0].imshow(c_show, cmap='gray', vmax=vmax[i],vmin=vmin[i])
        else:
            axes_comparison_plot[plot_idx][0].imshow(np.ones(shape=(300,300)), cmap='gray', vmax=1,vmin=0.99)


        axes_comparison_plot[plot_idx][1].imshow(real_M[i], cmap='gray', vmax=vmax[i],vmin=vmin[i]) 
        axes_comparison_plot[plot_idx][2].imshow(obsidian_dict['OBSIDIAN (No bias correction)'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])  
        axes_comparison_plot[plot_idx][3].imshow(predicted_images['unet_rician'][i], cmap='gray', vmax=vmax[i], vmin=vmin[i])
        axes_comparison_plot[plot_idx][4].imshow(obsidian_img, cmap='gray', vmax=vmax[i],vmin=vmin[i])  
        axes_comparison_plot[plot_idx][5].imshow(predicted_images['unet'][i],cmap='gray', vmax=vmax[i],vmin=vmin[i])
        axes_comparison_plot[plot_idx][6].imshow(predicted_images['attention_unet'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])  
        axes_comparison_plot[plot_idx][7].imshow(predicted_images['res_atten_unet'][i], cmap='gray', vmax=vmax[i],vmin=vmin[i])
    plt.subplots_adjust(left=0.021, right=1, top=0.91, bottom=0)
    plt.subplots_adjust(wspace=0.01, hspace=0.02)  # Reduce white space between images

def make_comparison_figure_reduced():

    for fit in  ['kurtosis', 'biexp', 'gamma']:#
        bvalues = [100.,1000.,1500.,2000.]
        patient = dual_patient_list[0]
        vindex = 12

        vmaxc  = np.array([1.2,0.4])
        comparison_figure_reduced(patient,bvalues, vindex=vindex, fitting_model=fit, new=True, dual_patients=True, second_pat=False, vmaxc=vmaxc)
        patient = dual_patient_list[1]
        vindex = 8
        vmaxc = np.array([1.3, 0.45])

        comparison_figure_reduced(patient,bvalues, vindex=vindex, fitting_model=fit, new=False, dual_patients=True, second_pat=True, vmaxc=vmaxc)
        path_to_save_png = os.path.join(FIGURE_SAVE_PATH, f'{fit}_dual.png')
        path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, f'{fit}_dual.pdf')
        plt.savefig(path_to_save_png, bbox_inches='tight')
        plt.savefig(path_to_save_pdf, bbox_inches='tight', dpi = 200)

def ADC_comparison_figure_avg_one_rice(patient,vindex,new = False, threshold = 0.05 ,dual = False):
    global biexp_file, gamma_file, kurtosis_file
    global fig_adc_comparison_plot, axes_adc_comparison_plot, outer_gs
    global true_M_biexp, true_M_gamma, true_M_kurtosis
    print(f'Loading...')
    factor = 1000
    base_result_path = result_paths
    test_data_path = patient_paths[patient]
    seg_file_name = patient + '_seg.nii.gz'

    test_seg_path = seg_paths[patient] +seg_file_name  #
    roi_file = nib.load(os.path.join(test_seg_path))  # , patient + '_seg.nii.gz'))
    roi_file = np.asanyarray(roi_file.dataobj).transpose(2, 1, 0)
    region_mask = roi_file == 4

    vmin = 0  # ADC_real.min()
    vmiddle = 2

    vmax = 3
    norm_custom = mcolors.Normalize(vmin=vmin, vmax=vmax)


    biexp_file = np.load(os.path.join(test_data_path, patient  + '_biexp.npz'), allow_pickle=True)['arr_0'][()]

    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]

    b100_index = [np.where(b_array == 100)[0][0]]
    b1000_index = [np.where(b_array == 1000)[0][0]]
    b100 = b_array[b100_index].numpy()/factor
    b1000 = b_array[b1000_index].numpy()/factor
    b_array = torch.tile(b_array, (3,))
    b = b_array.reshape(1, len(b_array), 1, 1)

    b00 = torch.tensor(biexp_file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)
    n_slices = b00.shape[0]
    b0_mean = np.mean((b00[:, 0][region_mask[:n_slices]]).numpy())
    b00 = b00[:,0,20:-20,:]
    print(f'Thresold pixel value: {b0_mean * threshold}')
    clinical_file = np.load(os.path.join(test_data_path,patient+'_clinical.npz'), allow_pickle=True)
    clinical_file = clinical_file['arr_0'][()]
    M_clinical = clinical_file['image_data'][:,:,20:-20,:]
    n_slices = min(M_clinical.shape[1],n_slices)

    bvals = clinical_file['bval_arr']
    if not isinstance(bvals, list):
        bvals = list(np.unique(bvals))

    M_clinical_b100 = M_clinical[bvals.index(100)]
    M_clinical_b1000 = M_clinical[bvals.index(1000)]


    if M_clinical.shape[1] > 22:
        M_clinical_b100 = M_clinical_b100[0:n_slices]
        M_clinical_b1000 = M_clinical_b1000[0:n_slices]

    M_clinical_b100[M_clinical_b100 < 1.] = 1
    M_clinical_b1000[M_clinical_b1000 < 1.] = 1

    ADC_clinical = -np.log(M_clinical_b1000 / M_clinical_b100) / (b1000 - b100)
    ADC_clinical = ADC_clinical[:n_slices]
    ADC_clinical[b00<b0_mean *threshold] = 0

    shape = (n_slices, b00.shape[1], b00.shape[2])#(1, b0.shape[2] - 40, b0.shape[3])
    num_diff = 3
    ADC_real_array = np.empty(shape=(num_diff, *shape))
    for i in range(num_diff):

        real_M_b100 = biexp_file['image']['3D'][:, :, 20:-20, :][:,i*20+b100_index[0]]#[i*20+b100_index[0]]
        real_M_b1000 = biexp_file['image']['3D'][:, :, 20:-20, :][:,i*20 + b1000_index[0]]#[i*20 + b1000_index[0]]


        real_M_b100[real_M_b100 <1] = 1
        real_M_b1000[real_M_b1000 <1] = 1
        ADC_real_array[i] = -np.log(real_M_b1000 / real_M_b100) / (b1000 - b100)
        ADC_real_array[i][b00<b0_mean *threshold] = 0
    ADC_real = np.mean(ADC_real_array,axis=0)

    models = [MAIN_METHOD+'/'+m for m in ['attention_unet', 'unet', 'res_atten_unet']] + [MAIN_METHOD+'_no_rice/unet']
    model_names_list = ['attention_unet', 'unet', 'res_atten_unet','unet_rician']
    fitting_models = ['biexp', 'kurtosis', 'gamma']

    fitting_name_dict = {'biexp': 'Biexponential',
                       'kurtosis': 'Kurtosis',
                       'gamma': 'Gamma'}
    ADC_obsidian = {'OBSIDIAN': {},'OBSIDIAN (No bias correction)': {}}
    B100_obsidian = {'OBSIDIAN': {},'OBSIDIAN (No bias correction)': {}}
    B1000_obsidian = {'OBSIDIAN': {},'OBSIDIAN (No bias correction)': {}}


    obs_names = ['OBSIDIAN', 'OBSIDIAN (No bias correction)']
    obs_file_name = ['','_nb']
    obs_result_name = ['3Dsig','3D']

    for idx,name in enumerate(obs_names):
        print(name)
        biexp_file=None
        gamma_file = None
        kurtosis_file = None
        ADC_obs_array = np.empty(shape=(num_diff, *shape))
        B100_obs_array = np.empty(shape=(num_diff, *shape))
        B1000_obs_array = np.empty(shape=(num_diff, *shape))
        for fitting_name in fitting_models:
            for i in range(num_diff):
                print(f'{fitting_name} {i}')
                if biexp_file is None:
                    biexp_file = np.load(os.path.join(test_data_path, patient + '_biexp'+obs_file_name[idx] + '.npz'), allow_pickle=True)['arr_0'][()]
                if gamma_file is None:
                    gamma_file = np.load(os.path.join(test_data_path, patient + '_gamma'+obs_file_name[idx]  + '.npz'), allow_pickle=True)['arr_0'][()]
                if kurtosis_file is None:
                    kurtosis_file = np.load(os.path.join(test_data_path, patient + '_kurtosis' +obs_file_name[idx] + '.npz'), allow_pickle=True)['arr_0'][()]
    

                if fitting_name == 'biexp':
                    true_d1 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+0])  ###Maybe expand to other direction
                    true_d2 = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+1])
                    true_f = torch.from_numpy(biexp_file['result'][obs_result_name[idx]][:, :, :, 3*i+2])
                    b0 = torch.tensor(biexp_file['result'][obs_result_name[idx]][:, :, :, -3], dtype=torch.float32).unsqueeze(dim=1)

                    true_M_biexp = b0 * bio_exp(d1=true_d1.unsqueeze(dim=1), d2=true_d2.unsqueeze(dim=1),
                                                f=true_f.unsqueeze(dim=1),
                                                b=b)
                if fitting_name == 'kurtosis':
                    true_d = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                    true_k = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                    b0 = torch.tensor(kurtosis_file['result'][obs_result_name[idx]][:, :, :, -3], dtype=torch.float32).unsqueeze(dim=1)

                    true_M_kurtosis = b0 * kurtosis(bval=b, D=true_d.unsqueeze(dim=1), K=true_k.unsqueeze(dim=1))
                if fitting_name == 'gamma':
                    true_k = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+0])  ###Maybe expand to other direction
                    true_theta = torch.from_numpy(gamma_file['result'][obs_result_name[idx]][:, :, :, 2*i+1])
                    b0 = torch.tensor(gamma_file['result'][obs_result_name[idx]][:, :, :, -3], dtype=torch.float32).unsqueeze(dim=1)

                    true_M_gamma = b0 * gamma(bval=b, K=true_k.unsqueeze(dim=1), theta=true_theta.unsqueeze(dim=1))

                if fitting_name == 'biexp':
                    obsidian_b100 = true_M_biexp[:, :, 20:-20, :][:,i*20+b100_index[0]].numpy()#[i*20+b100_index[0]].numpy() samma på alla
                    obsidian_b1000 = true_M_biexp[:, :, 20:-20, :][:,i*20+b1000_index[0]].numpy()
                if fitting_name == 'kurtosis':

                    obsidian_b100 = true_M_kurtosis[:, :, 20:-20, :][:,i*20+b100_index[0]].numpy()
                    obsidian_b1000 = true_M_kurtosis[:, :, 20:-20, :][:,i*20+b1000_index[0]].numpy()
                if fitting_name == 'gamma':

                    obsidian_b100 = true_M_gamma[:, :, 20:-20, :][:,i*20+b100_index[0]].numpy()
                    obsidian_b1000 = true_M_gamma[:, :, 20:-20, :][:,i*20+b1000_index[0]].numpy()

                obsidian_b100[obsidian_b100 <1] = 1
                obsidian_b1000[obsidian_b1000 <1] = 1
                ADC_obs_temp = -np.log(obsidian_b1000 / obsidian_b100) / (b1000 - b100)

                ADC_obs_array[i] = ADC_obs_temp[:shape[0]]
                ADC_obs_array[i][b00 < b0_mean * threshold] = 0

                B100_obs_array[i] = obsidian_b100[:shape[0]]
                B1000_obs_array[i] = obsidian_b1000[:shape[0]]
            ADC_obsidian[obs_names[idx]][fitting_name] = np.mean(ADC_obs_array, axis = 0)
            B100_obsidian[obs_names[idx]][fitting_name] = np.mean(B100_obs_array, axis = 0)
            B1000_obsidian[obs_names[idx]][fitting_name] = np.mean(B1000_obs_array, axis = 0)

    ADC_predicted = defaultdict(lambda: defaultdict(lambda: None))
    M_b100_predicted = defaultdict(lambda: defaultdict(lambda: None))
    M_b1000_predicted = defaultdict(lambda: defaultdict(lambda: None))
    ADC_predicted_array = np.empty(shape=(5,num_diff, *shape))
    M_b100_predicted_array = np.empty(shape=(5,num_diff, *shape))
    M_b1000_predicted_array = np.empty(shape=(5,num_diff, *shape))
    for mi,model in enumerate(models):
        model_name = model_names_list[mi]
        for fitting_name in fitting_models:

            for run_ind in range(5):
                result_path = base_result_path + model + '/' + fitting_name + '/' + patient + '/run_'+str(run_ind+1)+'/'
                M_file = np.load(f'{result_path}M.npy', allow_pickle=True)

                if not '3D' in MAIN_METHOD: M_file = np.concatenate((M_file[0:n_slices], M_file[n_slices:n_slices*2], M_file[n_slices*2:n_slices*3]), axis=1)
                for i in range(num_diff):
                    M = M_file[:, 20*i + 0:20*i + 20, :, :]

                    M_b100 = M[:,b100_index][:,0]#[b100_index] och ingen [:,0] för båda
                    M_b1000 = M[:,b1000_index][:,0]#[b1000_index]

                    M_b100[M_b100 < 1] = 1
                    M_b1000[M_b1000 < 1] = 1

                    M_b100_predicted_array[run_ind,i] = M_b100
                    M_b1000_predicted_array[run_ind,i] = M_b1000

                M_b100_predicted[model_name][fitting_name] = np.mean(np.mean(M_b100_predicted_array,axis = 0),axis =0)
                M_b1000_predicted[model_name][fitting_name] = np.mean(np.mean(M_b1000_predicted_array,axis = 0),axis =0)
                ADC_predicted[model_name][fitting_name] = -np.log(M_b1000_predicted[model_name][fitting_name]/M_b100_predicted[model_name][fitting_name]) / (b1000 - b100)
                ADC_predicted[model_name][fitting_name][b00 < b0_mean * threshold] = 0

    fontsize = 13

    if (fig_adc_comparison_plot is None or new) and not dual:
        fig_adc_comparison_plot, axes_adc_comparison_plot = plt.subplots(ncols=7, nrows=3, figsize=(18.2, 6.22))
        fig_adc_comparison_plot.canvas.manager.set_window_title(f'ADC map comparison_{threshold}')
        for ax in axes_adc_comparison_plot.flat:  # Flatten thpatiee 2D array of axes
            ax.axis("off")  # Hide x and y axes

    #end of test
    def add_grouped_grid(row, norm_custom, c_im, r_im, main_images, add_titles=False, group_label = None):
            # Side plots in column 0
            height_ratios = [ 1, 0.001, 1, 1 - 0.001]
            scale = 3.2 / 3  # or whatever scale you want to keep image sizes unchanged
            scaled_height_ratios = [h * scale for h in height_ratios]
            side_spec = gridspec.GridSpecFromSubplotSpec(
                4, 1,
                subplot_spec=outer_gs[row, 0],
                height_ratios=height_ratios
            )

            ax_label = fig_adc_comparison_plot.add_subplot(side_spec[0])
            ax_label.axis("off")
            ax_label.text(0, 1, group_label, fontsize=12, fontweight="bold", va="top", ha="left", transform=ax_label.transAxes)

            ax_side1 = plt.Subplot(fig_adc_comparison_plot, side_spec[0])
            ax_side1.imshow(c_im, cmap='gray', norm=norm_custom)
            ax_side1.set_title(f"Clinical ADC", fontsize=10)
            ax_side1.axis('off')
            fig_adc_comparison_plot.add_subplot(ax_side1)

            ax_side2 = plt.Subplot(fig_adc_comparison_plot, side_spec[2])
            ax_side2.imshow(r_im, cmap='gray', norm=norm_custom)
            ax_side2.set_title(f"Raw ADC", fontsize=10)
            ax_side2.axis('off')
            fig_adc_comparison_plot.add_subplot(ax_side2)

            # Main image grid in column 2
            main_spec = gridspec.GridSpecFromSubplotSpec(
                3, 7,
                subplot_spec=outer_gs[row, 2],
                hspace=0.02,
                wspace=0.02,
                width_ratios=[1, 1, 0.05, 1, 1, 1, 1]

            )

            titles = ['Direct fit', 'Standard U-Net (Direct)', 'OBSIDIAN', 'Standard U-Net', 'Attention U-Net',
                      'Residual Attention U-Net']
            labels = ['Biexponential', 'Kurtosis', 'Gamma']

            for i in range(3):
                for ji,j in enumerate([0,1,3,4,5,6]):
                    ax = fig_adc_comparison_plot.add_subplot(main_spec[i, j])
                    ax.imshow(main_images[i][ji], cmap='gray', norm=norm_custom)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if add_titles and i == 0:
                        ax.set_title(f"{titles[ji]}", fontsize=9, pad=2)

                    if j == 0:
                        ax.set_ylabel(f"{labels[i]}", fontsize=9, labelpad=2)
    if dual and new:
        fig_adc_comparison_plot = plt.figure(figsize=(12.9, 8.84))
        import matplotlib

        outer_gs = gridspec.GridSpec(
            3, 5,
            width_ratios=[1.55, 0.05, 7.7, 0.1, 0.2],  # added a small gap column (col=1)
            height_ratios=[1, 0.01, 1],
            wspace=0.01,
            hspace=0.05
        )


    elif not dual:
        plt.subplots_adjust(left=0.05, right=0.93, top=0.92, bottom=0)
        plt.subplots_adjust(wspace=0.02, hspace=0.03)
        im = axes_adc_comparison_plot[0, 0].imshow(ADC_clinical[vindex], cmap='gray', norm = norm_custom)  
        axes_adc_comparison_plot[0, 0].set_title(f'Clinical ADC')
        axes_adc_comparison_plot[0, 0].set(position=[0.00, 0.61, 0.1435, 0.309])
        axes_adc_comparison_plot[1, 0].imshow(ADC_real[vindex], cmap='gray', norm = norm_custom)  
        axes_adc_comparison_plot[1, 0].set_title(f'Raw ADC')
        axes_adc_comparison_plot[1, 0].set(position=[0.0, 0.2505, 0.1435, 0.309])
        plt.show()


        if True:
            cbar_ax = fig_adc_comparison_plot.add_axes([0.94, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
            cbar = fig_adc_comparison_plot.colorbar(im,cax=cbar_ax)#plt.cm.ScalarMappable(cmap='gray',norm=mcolors.Normalize(vmin=0, vmax=3)),

            cbar.set_ticks([vmin, vmax])

            cbar.set_label("ADC [\u00b5m\u00b2/ms]")

    rows = []

    for index, (fitting_key, obsidian_adc_map) in enumerate(ADC_obsidian['OBSIDIAN'].items()):
        plot_index = index
        rows.append([
            ADC_obsidian['OBSIDIAN (No bias correction)'][fitting_key][vindex],
            ADC_predicted['unet_rician'][fitting_key][vindex],
            obsidian_adc_map[vindex],
            ADC_predicted['unet'][fitting_key][vindex],
            ADC_predicted['attention_unet'][fitting_key][vindex],
            ADC_predicted['res_atten_unet'][fitting_key][vindex]
        ])
        if not dual:
            axes_adc_comparison_plot[plot_index, 1].imshow(ADC_obsidian['OBSIDIAN (No bias correction)'][fitting_key][vindex], cmap='gray', norm = norm_custom)  
            axes_adc_comparison_plot[plot_index, 1].text(-1.1, 0.5, f'{fitting_name_dict[fitting_key]}', fontsize = fontsize, rotation=90,ha='center', va='center',transform=axes_adc_comparison_plot[plot_index, 2].transAxes)
            axes_adc_comparison_plot[plot_index, 2].imshow(ADC_predicted['unet_rician'][fitting_key][vindex], cmap='gray', norm = norm_custom)  
            axes_adc_comparison_plot[plot_index, 3].imshow(obsidian_adc_map[vindex], cmap='gray', norm = norm_custom)  
            axes_adc_comparison_plot[plot_index, 4].imshow(ADC_predicted['unet'][fitting_key][vindex], cmap='gray', norm = norm_custom)  
            axes_adc_comparison_plot[plot_index, 5].imshow(ADC_predicted['attention_unet'][fitting_key][vindex], cmap='gray', norm = norm_custom)  
            axes_adc_comparison_plot[plot_index, 6].imshow(ADC_predicted['res_atten_unet'][fitting_key][vindex], cmap='gray', norm = norm_custom)  
            if index==0:
                #axes_adc_comparison_plot[0, 2].set_title(f'TRUE ADC')
                axes_adc_comparison_plot[0, 1].set_title(f'Direct fit')
                axes_adc_comparison_plot[0, 2].set_title(f'Standard U-Net\n(Direct)')
                axes_adc_comparison_plot[0, 3].set_title(f'OBSIDIAN')
                axes_adc_comparison_plot[0, 4].set_title(f'Standard U-Net')
                axes_adc_comparison_plot[0, 5].set_title(f'Attention U-Net')
                axes_adc_comparison_plot[0, 6].set_title(f'Residual Attention U-Net')
    if dual:
        if new:add_grouped_grid(0, norm_custom=norm_custom, c_im=ADC_clinical[vindex], r_im=ADC_real[vindex], main_images=rows,
                         add_titles=True, group_label='')
        else:
            add_grouped_grid(2, norm_custom=norm_custom, c_im=ADC_clinical[vindex], r_im=ADC_real[vindex], main_images=rows,
                         add_titles=False, group_label='')

        ax_line = fig_adc_comparison_plot.add_subplot(outer_gs[1, 2])
        ax_line.axhline(0.5, color='black', linewidth=1)
        ax_line.axis('off')

        cbar_gs = gridspec.GridSpecFromSubplotSpec(
            12, 1,
            subplot_spec=outer_gs[:, 4],
            hspace=0.1
        )

        cbar_ax = fig_adc_comparison_plot.add_subplot(cbar_gs[3:9])
        cbar = fig_adc_comparison_plot.colorbar(
            plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(0, 3)),
            cax=cbar_ax
        )
        cbar.set_ticks([0, 3])
        cbar_ax.set_ylabel("ADC [\u00b5m\u00b2/ms]", rotation=90, labelpad=0)
        plt.show()
        plt.subplots_adjust(left=0.01, right=0.95,
                            top=0.95, bottom=0.01)

def ADCK_comparison_figure_avg_one_rice(patient, vindex,parameter , new=False, threshold=0.05):
    global biexp_file, gamma_file, kurtosis_file
    global fig_adc_comparison_plot, axes_adc_comparison_plot, outer_gs
    global true_M_biexp, true_M_gamma, true_M_kurtosis
    print(f'Loading...')
    base_result_path = result_paths + MAIN_METHOD
    test_data_path = patient_paths[patient]
    seg_file_name = patient + '_seg.nii.gz'

    test_seg_path = seg_paths[patient] + seg_file_name
    roi_file = nib.load(os.path.join(test_seg_path))  # , patient + '_seg.nii.gz'))
    roi_file = np.asanyarray(roi_file.dataobj).transpose(2, 1, 0)
    region_mask = roi_file == 4


    vmax = 3 if parameter == 'adck' else 3
    vmin = 0 if parameter == 'adck' else 0
    norm_custom = mcolors.Normalize(vmin=vmin, vmax=vmax)

    biexp_file = np.load(os.path.join(test_data_path, patient + '_biexp.npz'), allow_pickle=True)['arr_0'][()]

    b_array = torch.linspace(0, 2000, steps=21)
    b_array = b_array[1:]



    b00 = torch.tensor(biexp_file['image_b0'][:, :, :], dtype=torch.float32).unsqueeze(dim=1)
    n_slices = b00.shape[0]
    b0_mean = np.mean((b00[:, 0][region_mask[:n_slices]]).numpy())
    b00 = b00[:, 0, 20:-20, :]
    print(f'Thresold pixel value: {b0_mean * threshold}')

    shape = (n_slices, b00.shape[1], b00.shape[2])  # (1, b0.shape[2] - 40, b0.shape[3])
    num_diff = 3

    models = ['unet', 'attention_unet','res_atten_unet']#'unet'
    fitting_models = ['kurtosis']

    ADC_obsidian = {'OBSIDIAN': {}, 'OBSIDIAN (No bias correction)': {}}

    obs_names = ['OBSIDIAN', 'OBSIDIAN (No bias correction)']
    obs_file_name = ['', '_nb']
    obs_result_name = ['3Dsig', '3D']

    for idx, name in enumerate(obs_names):
        print(name)
        biexp_file = None
        gamma_file = None
        kurtosis_file = None
        ADC_obs_array = np.empty(shape=(num_diff, *shape))
        for fitting_name in fitting_models:
            for i in range(num_diff):
                if kurtosis_file is None:
                        kurtosis_file = \
                        np.load(os.path.join(test_data_path, patient + '_kurtosis' + obs_file_name[idx] + '.npz'),
                                allow_pickle=True)['arr_0'][()]


                if fitting_name == 'kurtosis':
                    true_d = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :,
                                              2 * i + 0])  ###Maybe expand to other direction
                    true_k = torch.from_numpy(kurtosis_file['result'][obs_result_name[idx]][:, :, :,
                                              2 * i + 1])  ###Maybe expand to other direction

                else:
                    print('Not kurtosis?')



                if parameter == 'adck': ADC_obs_array[i] = true_d[:,20:-20].numpy()
                elif parameter == 'k': ADC_obs_array[i] = true_k[:,20:-20].numpy()
                ADC_obs_array[i][b00 < b0_mean * threshold] = 0


            ADC_obsidian[obs_names[idx]][fitting_name] = np.mean(ADC_obs_array, axis=0)


    ADC_predicted = defaultdict(lambda: defaultdict(lambda: None))

    for model_name in models:
        fitting_name =  fitting_models[0]
        ADC_predicted_array = np.empty(shape=(5, *shape))

        for run_ind in range(5):
            result_path = base_result_path  +'/'+ model_name + '/' + fitting_name + '/' + patient + '/run_' + str(
                run_ind + 1) + '/'
            ADCK_pred = np.load(f'{result_path}parameters.npy', allow_pickle=True)

     
            if '3D' in MAIN_METHOD:
                from pathlib import Path

                Path(os.path.join(FIGURE_SAVE_PATH, 'adcks')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(FIGURE_SAVE_PATH, 'ks')).mkdir(parents=True, exist_ok=True)
                if parameter == 'adck':
                    if model_name == 'none':
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'adcks', f'ADCK{run_ind + 1}{1}_{model_name}.png'),
                                   ADCK_pred[0, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'adcks', f'ADCK{run_ind + 1}{2}_{model_name}.png'),
                                   ADCK_pred[2, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'adcks', f'ADCK{run_ind + 1}{3}_{model_name}.png'),
                                   ADCK_pred[4, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                    ADCK_pred = np.mean(ADCK_pred[[0, 2, 4]], axis=0)

                elif parameter == 'k':
                    if model_name == 'none':
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'ks', f'K{run_ind + 1}{1}.png'), ADCK_pred[1, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'ks', f'K{run_ind + 1}{2}.png'), ADCK_pred[3, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                        plt.imsave(os.path.join(FIGURE_SAVE_PATH, 'ks', f'K{run_ind + 1}{3}.png'), ADCK_pred[5, vindex],
                                   cmap='gray', vmax=vmax, vmin=vmin)  
                    ADCK_pred = np.mean(ADCK_pred[[1, 3, 5]], axis=0)

            else:
                ADCK_pred = ADCK_pred[0]

            ADC_predicted_array[run_ind] = ADCK_pred
        ADC_predicted[model_name][fitting_name] = np.mean(ADC_predicted_array,axis =0)
        ADC_predicted[model_name][fitting_name][b00 < b0_mean * threshold] = 0

    ADC_predicted_norice = defaultdict(lambda: defaultdict(lambda: None))
    ADC_predicted_array = np.empty(shape=(5, *shape))
    for run_ind in range(5):
        result_path = base_result_path + '_no_rice/unet/' + fitting_name + '/' + patient + '/run_' + str(run_ind + 1) + '/'
        ADCK_pred = np.load(f'{result_path}parameters.npy', allow_pickle=True)


        if '3D' in MAIN_METHOD:
            if parameter == 'adck':

                ADCK_pred = np.mean(ADCK_pred[[0, 2, 4]], axis=0)

            elif parameter == 'k':

                ADCK_pred = np.mean(ADCK_pred[[1,3,5]], axis = 0)

        else:
            ADCK_pred = ADCK_pred[0]
        ADC_predicted_array[run_ind] = ADCK_pred
    ADC_predicted_norice['unet'][fitting_name] = np.mean(ADC_predicted_array,axis =0)
    ADC_predicted_norice['unet'][fitting_name][b00 < b0_mean * threshold] = 0



    if (fig_adc_comparison_plot is None or new):
        fig_adc_comparison_plot = plt.figure(figsize=(17.22, 2.7))
        gs = gridspec.GridSpec(nrows=1, ncols=7, figure=fig_adc_comparison_plot,
                               width_ratios=[1,1,1,1,1,1,0.1])
        axes_adc_comparison_plot = [fig_adc_comparison_plot.add_subplot(gs[0, j]) for j in range(6)]
        axes_adc_comparison_plot = np.array(axes_adc_comparison_plot)

        cbar_gs = gridspec.GridSpecFromSubplotSpec(
            12, 1,
            subplot_spec=gs[0, -1],
            hspace=0.05
        )

        cbar_ax = fig_adc_comparison_plot.add_subplot(cbar_gs[2:10])




        fig_adc_comparison_plot.canvas.manager.set_window_title(f'ADC map comparison_{threshold}')

        for ax in axes_adc_comparison_plot.flat:  # Flatten thpatiee 2D array of axes
            ax.axis("off")  # Hide x and y axes

    # end of test
    def add_grouped_grid(row, norm_custom, c_im, r_im, main_images, add_titles=False):
        main_spec = gridspec.GridSpecFromSubplotSpec(
            3, 6,
            subplot_spec=outer_gs[row, 2],
            hspace=0.02,
            wspace=0.02
        )

        titles = ['Direct fit', 'Standard U-Net\n(Direct)', 'OBSIDIAN', 'Standard U-Net', 'Attention U-Net',
                  'Residual Attention U-Net']
        labels = ['Biexponential', 'Kurtosis', 'Gamma']

        for i in range(3):
            for j in range(6):
                ax = fig_adc_comparison_plot.add_subplot(main_spec[i, j])
                ax.imshow(main_images[i][j], cmap='gray', norm=norm_custom)
                ax.set_xticks([])
                ax.set_yticks([])

                if add_titles and i == 0:
                    ax.set_title(f"{titles[j]}", fontsize=9, pad=2)

                if j == 0:
                    ax.set_ylabel(f"{labels[i]}", fontsize=9, labelpad=2)





    plt.subplots_adjust(left=0.00, right=0.85, top=0.98, bottom=0)
    plt.subplots_adjust(wspace=0.05, hspace=0.00)




    for index, (fitting_key, obsidian_adc_map) in enumerate(ADC_obsidian['OBSIDIAN'].items()):


        axes_adc_comparison_plot[0].imshow(ADC_obsidian['OBSIDIAN (No bias correction)'][fitting_key][vindex], cmap='gray',norm=norm_custom)
        axes_adc_comparison_plot[1].imshow(ADC_predicted_norice['unet'][fitting_key][vindex], cmap='gray',
                                              norm=norm_custom)
        axes_adc_comparison_plot[2].imshow(obsidian_adc_map[vindex], cmap='gray',
                                                       norm=norm_custom)
        im = axes_adc_comparison_plot[3].imshow(ADC_predicted['unet'][fitting_key][vindex], cmap='gray',
                                                       norm=norm_custom)
        axes_adc_comparison_plot[4].imshow(ADC_predicted['attention_unet'][fitting_key][vindex],
                                                       cmap='gray', norm=norm_custom)
        axes_adc_comparison_plot[5].imshow(ADC_predicted['res_atten_unet'][fitting_key][vindex],
                                                       cmap='gray', norm=norm_custom)
        if index == 0 and parameter != 'k':
            axes_adc_comparison_plot[0].set_title(f'Direct fit')
            axes_adc_comparison_plot[1].set_title(f'Standard U-Net (Direct)')
            axes_adc_comparison_plot[2].set_title(f'OBSIDIAN')
            axes_adc_comparison_plot[3].set_title(f'Standard U-Net')
            axes_adc_comparison_plot[4].set_title(f'Attention U-Net')
            axes_adc_comparison_plot[5].set_title(f'Residual Attention U-Net')
        if True:
            cbar = fig_adc_comparison_plot.colorbar(im,
                                                    cax=cbar_ax)  # plt.cm.ScalarMappable(cmap='gray',norm=mcolors.Normalize(vmin=0, vmax=3)),
            cbar.set_ticks([vmin, vmax])
            if parameter == 'adck':
                cbar.set_label(r"$\mathrm{ADC}_{\mathrm{K}}$ "+"[\u00b5m\u00b2/ms]", fontsize=12)
            elif parameter == 'k':
                cbar.set_label("K", fontsize=12)



    plt.show()

def make_ADCK_comparison_figure_avg_one_rice(parameter):
    patient = dual_patient_list[0]
    vindex = 12
    ADCK_comparison_figure_avg_one_rice(patient=patient, vindex=vindex,parameter=parameter, new=True, threshold=0.007)
    filename = 'ADCK' if parameter == 'adck' else 'K'
    path_to_save_png = os.path.join(FIGURE_SAVE_PATH, f'{filename}.png')
    path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, f'{filename}.pdf')
    plt.savefig(path_to_save_pdf, bbox_inches='tight', dpi = 200)
    plt.savefig(path_to_save_png, bbox_inches='tight')

def make_ADC_comparison_figure_avg_one_rice_dual():
    patient = dual_patient_list[0]
    vindex = 12
    ADC_comparison_figure_avg_one_rice(patient=patient,vindex= vindex,new = True, threshold=0.007, dual = True)
    patient = dual_patient_list[1]
    vindex = 8
    ADC_comparison_figure_avg_one_rice(patient=patient,vindex= vindex,new = False, threshold=0.007, dual = True)

    path_to_save_png = os.path.join(FIGURE_SAVE_PATH, f'ADC_dual.png')
    path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, f'ADC_dual.pdf')
    plt.savefig(path_to_save_png, bbox_inches='tight')
    plt.savefig(path_to_save_pdf, bbox_inches='tight', dpi = 300)

def make_dense_plot_ADC_obs_gs(patients, region_name, dict_runs_pixels, vmax=10, zoom=False):
    """:param keyword: Either 'clinical' or 'true'"""


    global fig_dense_plot_ADC, axes_dense_plot_ADC, table_ADC_print_toappend_flat

    fitting_name_dict = {'biexp': 'Biexponential',
                         'kurtosis': 'Kurtosis',
                         'gamma': 'Gamma'}
    model_name_dict = {'unet': 'Standard U-Net',
                       'attention_unet': 'Attention U-Net',
                       'res_atten_unet': 'Residual Attention U-Net'}


    fig_dense_plot_ADC = plt.figure(figsize=(10.8, 7.2))#7.2##CHANGE SIZE
    fig_dense_plot_ADC = plt.figure(figsize=(9.8, 8.2))#7.2##CHANGE SIZE
    fig_dense_plot_ADC.canvas.manager.set_window_title(
        f'ADC Comparison between AI OBSIDIAN and Clinical Data for {region_name}')


    outer_gs = gridspec.GridSpec(nrows=5, ncols=5, figure=fig_dense_plot_ADC,
                                 height_ratios=[1, 1, 1,0.05,0.1],
                                 width_ratios=[1,0.1, 1, 1, 1])  # Equal column widths

    if True:
        gs_col0 = gridspec.GridSpecFromSubplotSpec(nrows=4, ncols=1,
                                                   subplot_spec=outer_gs[:3, 0],
                                                   height_ratios=[0.0, 0.3, 1, 1 - 0.3])

        X_arrays = []
        Y_raw_arrays = []
        Y_clin_arrays = []
        for patient in patients:
            extracted_data = dict_runs_pixels['unet']['kurtosis'][patient][
                region_name]
            data_lower = {key.lower(): np.mean(np.array(value['pixels']), axis=0) for key, value in
                          extracted_data.items()}

            X = data_lower[f'true_adc']
            Y_raw = data_lower['raw_adc']
            Y_clin = data_lower['clinical_adc']
            X_arrays.append(X)
            Y_raw_arrays.append(Y_raw)
            Y_clin_arrays.append(Y_clin)

        X = np.concatenate(X_arrays)

        Y_raw = np.concatenate(Y_raw_arrays)
        Y_clin = np.concatenate(Y_clin_arrays)

        xy_raw = np.vstack([X, Y_raw])
        density_raw = gaussian_kde(xy_raw)(xy_raw)

        if zoom:
            min_true = 0
            max_true = 3
        else:
            min_true = 0
            max_true = 1.05 * 4  # np.max([X.max(),Y_ai.max(),Y_true.max()])

        mae_raw = np.mean(np.abs(Y_raw - X))
        # Top and bottom subplots in column 0
        if False:
            ax_col0_top = fig_dense_plot_ADC.add_subplot(gs_col0[0, 0])
            ax_col0_top.scatter(X, Y_raw, c=density_raw, s=4, marker='o', vmax=vmax, vmin=0)  
            ax_col0_top.set_xlabel('OBSIDIAN')
            ax_col0_top.set_ylabel('Estimated ADC')
            ax_col0_top.plot([min_true, max_true], [min_true, max_true], color='red', linestyle='--')
            ax_col0_top.set_title(f'Raw')
            ax_col0_top.set_xlim(min_true, max_true)  # Set x-axis limits (min, max)
            ax_col0_top.set_ylim(min_true, max_true)  # Set y-axis limits (min, max)
            ax_col0_top.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax_col0_top.legend(title=f'MAE: {mae_raw:.3f}', fontsize=4, loc='lower right')
            ax_col0_top.set_xticks([0,1,2,3])
            ax_col0_top.set_yticks([0,1,2,3])
            ax_col0_top.set_aspect('equal', adjustable='box')
            ax_col0_top.axis("on")



        if False:

            xy_raw = np.vstack([X, Y_clin])
            density_raw = gaussian_kde(xy_raw)(xy_raw)
            mae_raw = np.mean(np.abs(Y_clin - X))
            ax_col0_bottom = fig_dense_plot_ADC.add_subplot(gs_col0[2, 0])

            ax_col0_bottom.scatter(X, Y_clin, c=density_raw, s=4, marker='o', vmax=vmax, vmin=0)  
            ax_col0_bottom.set_xlabel('')
            ax_col0_bottom.set_ylabel('')
            ax_col0_bottom.plot([min_true, max_true], [min_true, max_true], color='red', linestyle='--')
            ax_col0_bottom.set_title(f'Clinical')
            ax_col0_bottom.set_xlim(min_true, max_true)  # Set x-axis limits (min, max)
            ax_col0_bottom.set_ylim(min_true, max_true)  # Set y-axis limits (min, max)
            ax_col0_bottom.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax_col0_bottom.set_xticks([0, 1, 2, 3])
            ax_col0_bottom.set_yticks([0, 1, 2, 3])
            ax_col0_bottom.legend(title=f'MAE: {mae_raw:.3f}', fontsize=4, loc='lower right')
            ax_col0_bottom.axis("on")

    if zoom:
        min_true = 0
        max_true = 3
    else:
        min_true = 0
        max_true = 1.05 * 4

    for model_index, (model_key, model_name) in enumerate(model_name_dict.items()):
        model_index = model_index+2
        for fit_index, (fitting_key, fitting_name) in enumerate(fitting_name_dict.items()):

            Y_ai_arrays  =[]
            Y_true_arrays  =[]
            for patient in patients:
                print(patient)
                extracted_data = dict_runs_pixels[model_key][fitting_key][patient][region_name]
                data_lower = {key.lower(): np.mean(np.array(value['pixels']), axis=0) for key, value in
                              extracted_data.items()}
                Y_ai = data_lower['adc']
                Y_true = data_lower[f'true_adc']
                Y_ai_arrays.append(Y_ai)
                Y_true_arrays.append(Y_true)
            Y_ai = np.concatenate(Y_ai_arrays)
            Y_true = np.concatenate(Y_true_arrays)

            xy_ai = np.vstack([Y_true, Y_ai])
            density_ai = gaussian_kde(xy_ai)(xy_ai)

            print(f'{np.max([density_ai])}')
            axsc = fig_dense_plot_ADC.add_subplot(outer_gs[fit_index, model_index])
            sc = axsc.scatter(Y_true, Y_ai, c=density_ai, s=4, marker='o', vmax=vmax,
                                                                vmin=0)  

            mae_ai = np.mean(np.abs(Y_ai - Y_true))

            axsc.plot([min_true, max_true], [min_true, max_true], color='red',
                                                             linestyle='--')

            if model_index == 2:
                axsc.set_ylabel(f'{fitting_name}')
            if fit_index == 0:
                axsc.set_title(f'{model_name}')

            axsc.set_xlim(min_true, max_true)  # Set x-axis limits (min, max)
            axsc.set_ylim(min_true, max_true)  # Set y-axis limits (min, max)

            axsc.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            axsc.set_xticks([0, 1, 2, 3])
            axsc.set_yticks([0, 1, 2, 3])
            axsc.legend(title=f'MAE: {mae_ai:.3f}', fontsize=4,loc='lower right')
            axsc.set_aspect('equal', adjustable='box')

            axsc.axis("on")
            if model_index == 2 and False:
                Y_true = data_lower[f'true_adc']

                xy_true = np.vstack([X, Y_true])
                density_true = gaussian_kde(xy_true)(xy_true)

                axes_dense_plot_ADC[fit_index, -1].scatter(X, Y_true, c=density_true, s=4, marker='o', vmax=vmax,
                                                           vmin=0)  

                mae_true = np.mean(np.abs(Y_true - X))
                axes_dense_plot_ADC[fit_index, -1].plot([min_true, max_true], [min_true, max_true], color='red',
                                                        linestyle='--')

                if fit_index == 0: axes_dense_plot_ADC[fit_index, -1].set_title(f'OBSIDIAN')

                axes_dense_plot_ADC[fit_index, -1].set_xlim(min_true, max_true)  # Set x-axis limits (min, max)
                axes_dense_plot_ADC[fit_index, -1].set_ylim(min_true, max_true)  # Set y-axis limits (min, max)
                axes_dense_plot_ADC[fit_index, -1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

                axes_dense_plot_ADC[fit_index, -1].legend(title=f'MAE: {mae_true:.3f}', fontsize=4, loc='lower right')
                axes_dense_plot_ADC[fit_index, -1].axis("on")

    cbar_gs = gridspec.GridSpecFromSubplotSpec(
        1, 12,
        subplot_spec=outer_gs[-1,2:],
        hspace=0.1
    )
    cbar_ax = fig_dense_plot_ADC.add_subplot(cbar_gs[2:10])

    cbar = fig_dense_plot_ADC.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.03*vmax, 0.97*vmax])  # positions where colors change
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Pixel density', labelpad=-15)


    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)  # Reduce white space between images
    plt.subplots_adjust(wspace=0.25, hspace=0.2)  # Reduce white space between images
    fig_dense_plot_ADC.text(0.665, 0.125, "OBSIDIAN", ha='center', va='center', fontsize=12)  # X below all subplots
    fig_dense_plot_ADC.text(0.30, 0.56, "Estimated ADC", ha='center', va='center', rotation='vertical',
             fontsize=12)  # Y left of all subplots


    plt.show()
    path_to_save_png = os.path.join(FIGURE_SAVE_PATH, f'AI_vs_OBSIDIAN.png')
    path_to_save_pdf = os.path.join(FIGURE_SAVE_PATH, f'AI_vs_OBSIDIAN.pdf')#Very big file, laggy
    plt.savefig(path_to_save_png, bbox_inches='tight', dpi = 200)
    #plt.savefig(path_to_save_pdf, bbox_inches='tight')

def ADCTable(model_name,model,fitting_name,MAIN_METHOD,first_time = False, print_each_run = False, print_each_pat = False,
             standard_error = False, custom_patient = None, simple_std = False,new_for_obs=True):
    # 0     0    0    0        0  0  0    "Clear Label"
    # 1   255    0    0        1  1  1    "Lesion_peri"
    # 2     0  255    0        1  1  1    "Lesion_cent"
    # 3     0    0  255        1  1  1    "Healthy_peri"
    # 4   255  255    0        1  1  1    "Healthy_cent"
    # 5     0  255  255        1  1  1    "Lesion_peri_certain"
    # 6   255    0  255        1  1  1    "Lesion_cent_certain"
    global table_vs_measured_loss_ADC
    global table_ADC_print_toappend,table_ADC_print_toappend_flat
    global dict_runs_pixels_ADC
    factor = 1000

    if custom_patient is not None:
        patient_name_list = custom_patient
    else:
        assert False, 'Give a custom patient list'
    dict_patients_means = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"mean": None, "SE": None})))
    regions = ['Lesion\\_peri', 'Lesion\\_cent', 'Healthy\\_peri', 'Healthy\\_cent', 'Lesion\\_peri\\_certain',
               'Lesion_cent\\_certain']

    for patient in patient_name_list:
        folder_name = patient
        test_data_path = patient_paths[patient]  # Your npy files for eac patient and fittin model
        seg_file_name = patient+'_seg.nii.gz'
        seg_file_name_clin = patient+'_clinical_seg.nii.gz'
        file_name = patient+'_'+fitting_name+'.npz'

        base_path = os.path.join(result_paths+model,fitting_name, folder_name)#contain run_1 to run_5 folder

        test_seg_path = os.path.join(seg_paths[patient],seg_file_name)
        test_seg_path_clin = os.path.join(seg_paths[patient],seg_file_name_clin)

        file = np.load(os.path.join(test_data_path, file_name), allow_pickle=True)['arr_0'][()]
        print(f'loaded {os.path.join(test_data_path, file_name)}')

        if os.path.exists(test_seg_path_clin):
            roi_file_clin = nib.load(seg_path_clinical)  # , patient + '_seg.nii.gz'))
            roi_file_clin = np.asanyarray(roi_file_clin.dataobj).transpose(2, 1, 0)
            roi_file_clin = roi_file_clin[:, 20:-20, :]
        else:
            roi_file_clin  =None
        roi_file = nib.load(os.path.join(test_seg_path))#, patient + '_seg.nii.gz'))
        roi_file = np.asanyarray(roi_file.dataobj).transpose(2, 1, 0)
        roi_file = roi_file[:, 20:-20, :]

        b_array = torch.linspace(0, 2000, steps=21)
        b_array = b_array[1:]

        b100_index = np.where(b_array == 100)[0][0]
        b1000_index = np.where(b_array == 1000)[0][0]
        b100 = b_array[b100_index].numpy()
        b1000 = b_array[b1000_index].numpy()
        b = b_array.reshape(1, len(b_array), 1, 1)

        b0 = torch.tensor(file['result']['3Dsig'][:, :, :,-3], dtype=torch.float32).unsqueeze(dim=1)
        sigma_snr = file['result']['3Dsig'][:, :, :,-2]
        sigma_snr[sigma_snr<0.001] = 0.001
        b0_snr = file['result']['3Dsig'][:, :, :,-3]
        true_snr = (b0_snr/sigma_snr)[:,20:-20]

        shape = (b0.shape[0],b0.shape[2]-40, b0.shape[3])
        num_diff = 3

        ADC_true_array = np.empty(shape = (num_diff,*shape))#3,22,200,24
        ADCK_true_array = np.empty(shape = (num_diff,*shape))
        K_true_array = np.empty(shape = (num_diff,*shape))

        for i in range(num_diff):
            if fitting_name == 'biexp':
                true_d1 = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*3+ 0])  ###Maybe expand to other direction
                true_d2 = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*3+ 1])
                true_f =  torch.from_numpy(file['result']['3Dsig'][:, :, :, i*3+ 2])
                true_M = b0 * bio_exp(true_d1.unsqueeze(dim=1), true_d2.unsqueeze(dim=1), true_f.unsqueeze(dim=1), b)
    #
            elif fitting_name == 'kurtosis':
                true_d = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*2+ 0])  ###Maybe expand to other direction
                ADCK_true_array[i] = true_d[:,20:-20].numpy()
                true_k = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*2+ 1])
                K_true_array[i] = true_k[:,20:-20].numpy()
                true_M = b0 * kurtosis(bval= b, D=true_d.unsqueeze(dim=1), K = true_k.unsqueeze(dim=1))
    #
            elif fitting_name == 'gamma':
                true_k = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*2+ 0])  ###Maybe expand to other direction
                true_theta = torch.from_numpy(file['result']['3Dsig'][:, :, :, i*2+ 1])
                true_M = b0 * gamma(bval=b, K=true_k.unsqueeze(dim=1), theta=true_theta.unsqueeze(dim=1))

            roi_image = roi_file
            true_M = true_M.numpy()
            true_M_b100 = true_M[:,b100_index,20:-20]
            true_M_b1000 =  true_M[:,b1000_index,20:-20]
            true_M_b100[true_M_b100 == 0.] = 1
            true_M_b1000[true_M_b1000 == 0.] = 1
            ADC_true_array[i] = -factor*np.log(true_M_b1000 / true_M_b100) / (b1000 - b100)
        ADC_true = np.mean(ADC_true_array, axis=0)
        if fitting_name == 'kurtosis': 
            ADCK_true = np.mean(ADCK_true_array, axis=0)
            K_true = np.mean(K_true_array, axis=0)
        else: 
            ADCK_true = None
            K_true = None



        ADC_raw_array = np.empty(shape=(num_diff, *shape))
        for i in range(num_diff):
            raw_image = file['image']['3D'][:, i*20+0:i*20+20, 20:-20, :]
            raw_image_b100 = raw_image[:,b100_index]
            raw_image_b1000 =raw_image[:,b1000_index]
            raw_image_b100[raw_image_b100==0.] = 1
            raw_image_b1000[raw_image_b1000==0.] = 1
            ADC_raw_array[i] = -factor * np.log(raw_image_b1000 / raw_image_b100) / (b1000 - b100)

        ADC_raw = np.mean(ADC_raw_array, axis=0)


        clinical_file = np.load(os.path.join(test_data_path, patient + '_clinical.npz'), allow_pickle=True)
        clinical_file = clinical_file['arr_0'][()]
        M_clinical = clinical_file['image_data'][:, :, 20:-20, :]


        bvals = clinical_file['bval_arr']
        if not isinstance(bvals,list):
            bvals = list(np.unique(bvals))


        M_clinical_b100 = M_clinical[bvals.index(100)]
        M_clinical_b1000 = M_clinical[bvals.index(1000)]
        M_clinical_b100[M_clinical_b100 == 0.] = 1
        M_clinical_b1000[M_clinical_b1000 == 0.] = 1
        ADC_clinical = -factor*np.log(M_clinical_b1000 / M_clinical_b100) / (b1000 - b100)

        regions_idx = np.unique(roi_image)  # roiimage (22,200,240)

        dict_runs_means = defaultdict(lambda: defaultdict(lambda: {"means": [], "SE": []}))


        for index,run in enumerate(np.sort(os.listdir(base_path))):

            base_result_path = os.path.join(base_path,run)

            M_file = np.load(f'{base_result_path}/M.npy', allow_pickle=True)
            par_file = np.load(f'{base_result_path}/parameters.npy', allow_pickle=True)
            n_slices = M_file.shape[0]//3

            if not '3D' in MAIN_METHOD:
                M_file = np.concatenate((M_file[0:n_slices], M_file[n_slices:n_slices*2], M_file[n_slices*2:n_slices*3]), axis=1)#[:, 0:20, :, :]##may need to change if you want specific dir##
                ADCK_predicted = par_file[0]
                K_predicted = par_file[1]
            else:
                ADCK_predicted = np.mean(par_file[[0, 2, 4]], axis=0)
                K_predicted = np.mean(par_file[[1, 3, 5]], axis=0)
            print(f'ADCK_predicted shape {ADCK_predicted.shape}')
            ADC_predicted_array = np.empty(shape=(num_diff,*shape))
            for i in range(num_diff):
                M = M_file[:, 20 * i + 0:20 * i + 20, :, :]
                M_b100 = M[:,b100_index]
                M_b1000 = M[:,b1000_index]
                M_b100[M_b100 == 0.] = 1
                M_b1000[M_b1000 == 0.] = 1
                ADC_predicted_array[i] = -factor*np.log(M_b1000 / M_b100) / (b1000 - b100)
            ADC_M = np.mean(ADC_predicted_array, axis=0)



            measured_loss = np.load(f'{base_result_path}/loss.npy', allow_pickle=True)
            accumulative_loss = 0
            accumulative_loss_clinical = 0
            num_regions = 0
            results = []
            dict_runs_region_means   = defaultdict(lambda: defaultdict(lambda: {"means": None, "SE": None}))

            for region in regions_idx:
                if region == 0:
                    continue  # Skip background or unlabeled regions if 0 is the background
                num_regions+=1
                region_name = regions[region-1]
                region_mask = roi_image == region
                min_slice = min(region_mask.shape[0], ADC_clinical.shape[0], ADC_true.shape[0], ADC_raw.shape[0], ADC_M.shape[0])

                if roi_file_clin is None:
                    c_region_mask = region_mask
                else:
                    c_region_mask = roi_file_clin == region


                if 1==1:
                    region_mask = region_mask[:min_slice]
                    if roi_file_clin is not None:c_region_mask = c_region_mask[:min_slice]
                    ADC_clinical = ADC_clinical[:min_slice]
                    ADC_true = ADC_true[:min_slice]
                    if ADCK_true is not None: ADCK_true = ADCK_true[:min_slice]
                    if K_true is not None: K_true = K_true[:min_slice]
                    ADC_raw = ADC_raw[:min_slice]
                    ADC_M = ADC_M[:min_slice]
                    true_snr = true_snr[:min_slice]

                param_mean = np.mean(ADC_M[region_mask])
                dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['ADC']['pixels'].append(ADC_M[region_mask])

                if standard_error: param_std = np.std(ADC_M[region_mask])/np.sqrt(np.size(ADC_M[region_mask]))
                else: param_std = np.std(ADC_M[region_mask])
                dict_runs_means[region_name]['ADC']['means'].append(param_mean)
                dict_runs_means[region_name]['ADC']['SE'].append(param_std if not simple_std else 0)

                dict_runs_region_means[region_name]['ADC']['means'] = param_mean
                dict_runs_region_means[region_name]['ADC']['SE'] = param_std if not simple_std else 0


                true_param_mean = np.mean(ADC_true[region_mask])
                dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['true_ADC']['pixels'].append(
                    ADC_true[region_mask])

                if ADCK_true is not None:
                    ADCKtrue_param_mean = np.mean(ADCK_true[region_mask])
                    Ktrue_param_mean = np.mean(K_true[region_mask])
                    dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['Ktrue_ADC']['pixels'].append(
                        ADCK_true[region_mask])
                    dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['KKtrue_ADC']['pixels'].append(
                        K_true[region_mask])
                    ADCKparam_mean = np.mean(ADCK_predicted[region_mask])
                    Kparam_mean = np.mean(K_predicted[region_mask])
                    dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['K_ADC']['pixels'].append(
                        ADCK_predicted[region_mask])
                    dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['KK_ADC']['pixels'].append(
                        K_predicted[region_mask])
                    if standard_error:
                        ADCKparam_std = np.std(ADCK_predicted[region_mask]) / np.sqrt(np.size(ADCK_predicted[region_mask]))
                        Kparam_std = np.std(K_predicted[region_mask]) / np.sqrt(np.size(K_predicted[region_mask]))
                    else:
                        ADCKparam_std = np.std(ADCK_predicted[region_mask])
                        Kparam_std = np.std(K_predicted[region_mask])
                    dict_runs_means[region_name]['K_ADC']['means'].append(ADCKparam_mean)
                    dict_runs_means[region_name]['KK_ADC']['means'].append(Kparam_mean)
                    dict_runs_means[region_name]['K_ADC']['SE'].append(ADCKparam_std if not simple_std else 0)
                    dict_runs_means[region_name]['KK_ADC']['SE'].append(Kparam_std if not simple_std else 0)
                    dict_runs_region_means[region_name]['ADC']['means'] = ADCKparam_mean
                    dict_runs_region_means[region_name]['K']['means'] = Kparam_mean
                    dict_runs_region_means[region_name]['ADC']['SE'] = ADCKparam_std if not simple_std else 0
                    dict_runs_region_means[region_name]['K']['SE'] = Kparam_std if not simple_std else 0

                true_snr_mean = np.mean(true_snr[region_mask])


                if standard_error: true_param_std = np.std(ADC_true[region_mask]) / np.sqrt(np.size(ADC_true[region_mask]))
                else: true_param_std = np.std(ADC_true[region_mask])

                if ADCK_true is not None:
                    if standard_error:
                        ADCKtrue_param_std = np.std(ADCK_true[region_mask]) / np.sqrt(np.size(ADCK_true[region_mask]))
                        Ktrue_param_std = np.std(K_true[region_mask]) / np.sqrt(np.size(K_true[region_mask]))
                    else:
                        ADCKtrue_param_std = np.std(ADCK_true[region_mask])
                        Ktrue_param_std = np.std(K_true[region_mask])


                if standard_error:
                    true_snr_std = np.std(true_snr[region_mask]) / np.sqrt(np.size(true_snr[region_mask]))
                else:
                    true_snr_std = np.std(true_snr[region_mask])


                clinical_param_mean = np.mean(ADC_clinical[c_region_mask])
                dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['clinical_ADC']['pixels'].append(
                    ADC_clinical[c_region_mask])

                raw_param_mean = np.mean(ADC_raw[region_mask])
                dict_runs_pixels_ADC[model_name][fitting_name][patient][region_name]['raw_ADC']['pixels'].append(ADC_raw[region_mask])


                if standard_error: clinical_param_std = np.std(ADC_clinical[c_region_mask]) / np.sqrt(np.size(ADC_clinical[c_region_mask]))
                else: clinical_param_std = np.std(ADC_clinical[c_region_mask])

                if standard_error: raw_param_std = np.std(ADC_raw[region_mask]) / np.sqrt(np.size(ADC_raw[region_mask]))
                else: raw_param_std = np.std(ADC_raw[region_mask])


                if index == 0:
                    dict_runs_means[region_name]['true_ADC']['means'].append(true_param_mean)
                    dict_runs_means[region_name]['true_ADC']['SE'].append(true_param_std if not simple_std else 0)
                    if ADCK_true is not None:
                        dict_runs_means[region_name]['Ktrue_ADC']['means'].append(ADCKtrue_param_mean)
                        dict_runs_means[region_name]['KKtrue_ADC']['means'].append(Ktrue_param_mean)
                        dict_runs_means[region_name]['Ktrue_ADC']['SE'].append(ADCKtrue_param_std if not simple_std else 0)
                        dict_runs_means[region_name]['KKtrue_ADC']['SE'].append(Ktrue_param_std if not simple_std else 0)
                    dict_runs_means[region_name]['true_SNR']['means'].append(true_snr_mean)
                    dict_runs_means[region_name]['true_SNR']['SE'].append(true_snr_std if not simple_std else 0)
                    dict_runs_means[region_name]['clinical_ADC']['means'].append(clinical_param_mean)
                    dict_runs_means[region_name]['clinical_ADC']['SE'].append(clinical_param_std if not simple_std else 0 )
                    dict_runs_means[region_name]['raw_ADC']['means'].append(raw_param_mean)
                    dict_runs_means[region_name]['raw_ADC']['SE'].append(raw_param_std if not simple_std else 0)

                dict_runs_region_means[region_name]['true_ADC']['means'] = true_param_mean
                dict_runs_region_means[region_name]['true_ADC']['SE'] = true_param_std if not simple_std else 0
                if ADCK_true is not None:
                    dict_runs_region_means[region_name]['Ktrue_ADC']['means'] = ADCKtrue_param_mean
                    dict_runs_region_means[region_name]['KKtrue_ADC']['means'] = Ktrue_param_mean
                    dict_runs_region_means[region_name]['Ktrue_ADC']['SE'] = ADCKtrue_param_std if not simple_std else 0
                    dict_runs_region_means[region_name]['KKtrue_ADC']['SE'] = Ktrue_param_std if not simple_std else 0
                dict_runs_region_means[region_name]['true_SNR']['means'] = true_snr_mean
                dict_runs_region_means[region_name]['true_SNR']['SE'] = true_snr_std if not simple_std else 0
                dict_runs_region_means[region_name]['clinical_ADC']['means'] = clinical_param_mean
                dict_runs_region_means[region_name]['clinical_ADC']['SE'] = clinical_param_std if not simple_std else 0

                result_dict_to_append = {'Region': region_name}
                for par_key, param in dict_runs_region_means[region_name].items():
                    if 'true' in par_key or 'clinical' in par_key or 'K' in par_key:
                        continue
                    predicted_par_value = dict_runs_region_means[region_name][par_key]['means']
                    predicted_par_std = dict_runs_region_means[region_name][par_key]['SE']

                    true_key = 'true_'+par_key
                    clinical_key = 'clinical_'+par_key
                    snr_key = 'true_SNR'

                    true_par_value = dict_runs_region_means[region_name][true_key]['means']
                    true_par_std = dict_runs_region_means[region_name][true_key]['SE']

                    true_snr_value = dict_runs_region_means[region_name][snr_key]['means']
                    true_snr_std = dict_runs_region_means[region_name][snr_key]['SE']

                    clinical_par_value = dict_runs_region_means[region_name][clinical_key]['means']
                    clinical_par_std = dict_runs_region_means[region_name][clinical_key]['SE']

                    accumulative_loss += np.abs(true_par_value-predicted_par_value)
                    accumulative_loss_clinical += np.abs(clinical_par_value-predicted_par_value)
                    result_dict_to_append[f"Predicted {true_key.split('_')[1]}"] = f"{predicted_par_value:.3f}±{predicted_par_std:.3f}"
                    result_dict_to_append[f"True {true_key.split('_')[1]}"] = f"{true_par_value:.3f}±{true_par_std:.3f}"
                    result_dict_to_append[f"Clinical {true_key.split('_')[1]}"] = f"{clinical_par_value:.3f}±{clinical_par_std:.3f}"
                    result_dict_to_append[f"True SNR"] = f"{true_snr_value:.3f}±{true_snr_std:.3f}"


                results.append(result_dict_to_append)

            table_vs_measured_loss_ADC[patient][model_name][fitting_name]['true_loss'].append(accumulative_loss/num_regions)
            table_vs_measured_loss_ADC[patient][model_name][fitting_name]['clinical_loss'].append(accumulative_loss_clinical/num_regions)
            table_vs_measured_loss_ADC[patient][model_name][fitting_name]['measured_loss'].append(measured_loss)
            # Print the table with borders
            if print_each_run:
                results_df = pd.DataFrame(results).to_latex(index = False)
                results_df = results_df.replace(r"\toprule", "").replace(r"\midrule", "").replace(r"\bottomrule","").strip()
                new_lines = []
                lines = results_df.split("\n")
                for line in lines:
                    new_lines.append(line)
                    if "&" in line or 'begin' in line:  # This ensures we only add \hline after data rows
                        new_lines.append(r"\hline")

                results_df = "\n".join(new_lines)
                print(f'True loss: {accumulative_loss:.3f}\\\\Clinical loss: {accumulative_loss_clinical:.3f}\\\\\n {results_df}\n')


        # Mean of means and SE
        for region_key,dict_regions in dict_runs_means.items():

            for param_key,dict_param  in dict_regions.items():
                if not dict_param['means']:
                    break

                Mean,SE_Mean = calc_mean(mean_array = dict_param['means'],se_array = dict_param['SE'],pooled=True,standard_error=standard_error, no_std = False)
                dict_patients_means[patient][region_key][param_key]['mean'] = Mean
                dict_patients_means[patient][region_key][param_key]['SE'] = SE_Mean

    merged_data = defaultdict(lambda: defaultdict(lambda: {"mean": [], "SE": []}))

    # Merge patients
    for patient, regions in dict_patients_means.items():
        for region, parameters in regions.items():
            for param, values in parameters.items():
                merged_data[region][param]["mean"].append(values["mean"])
                merged_data[region][param]["SE"].append(values["SE"])


    # Convert defaultdict to regular dict (optional)
    merged_data = {region: {param: dict(values) for param, values in params.items()} for region, params in merged_data.items()}

    average_all_pat = defaultdict(lambda: defaultdict(lambda: {"means": None, "SE":None, "p-adc": 1.0, "p-adck": 1.0, "p-k": 1.0}))
    #T test
    ttestList = ['ADC', 'clinical_ADC','raw_ADC']
    for region_key, dict_regions in merged_data.items():

        for param_key, dict_param in dict_regions.items():
            if not dict_param['mean']:
                break

            if param_key in ttestList and len(dict_regions['true_ADC']['mean'])>1:
                TTestresult = ttest_ind(dict_regions['true_ADC']['mean'], dict_param['mean'])
                average_all_pat[region_key][param_key]['p-adc'] = TTestresult.pvalue
            elif fitting_name == 'kurtosis' and len(dict_regions['Ktrue_ADC']['mean'])>1:
                TTestresult = ttest_ind(dict_regions['Ktrue_ADC']['mean'], dict_regions['K_ADC']['mean'])
                average_all_pat[region_key]['K_ADC']['p-adck'] = TTestresult.pvalue
                TTestresult = ttest_ind(dict_regions['KKtrue_ADC']['mean'], dict_regions['KK_ADC']['mean'])
                average_all_pat[region_key]['KK_ADC']['p-k'] = TTestresult.pvalue

            Mean, SE_Mean = calc_mean(mean_array=dict_param['mean'], se_array=dict_param['SE'],standard_error=standard_error, normal_std = simple_std)
            average_all_pat[region_key][param_key]['means'] = Mean
            average_all_pat[region_key][param_key]['SE'] = SE_Mean



    # List to store formatted rows
    for pat_key,region_dict in dict_patients_means.items():
        # Loop through each region in the dictionary to make table
        table_rows = []
        accumulative_loss=0
        accumulative_loss_clinical=0
        for region, params in region_dict.items():
            row = {'Region': region}
            not_none = True
            for param, values in params.items():
                print(params.keys())
                if 'true' in param or 'clinical' in param or 'raw' in param or 'K_' in param:
                    continue
                mean = params[param]['mean']
                se = params[param]['SE']

                true_key = [s for s in params.keys() if 'true' in s and s.split('_')[1].lower() == param.lower()][0]
                true_mean = params[true_key]['mean']
                true_se = params[true_key]['SE']

                true_snr_mean = params['true_SNR']['mean']
                true_snr_se = params['true_SNR']['SE']


                clinical_key = [s for s in params.keys() if 'clinical' in s and s.split('_')[1].lower() == param.lower()][0]
                clinical_mean = params[clinical_key]['mean']
                clinical_se = params[clinical_key]['SE']

                raw_key = [s for s in params.keys() if 'raw' in s and s.split('_')[1].lower() == param.lower()][0]
                raw_mean = params[raw_key]['mean']
                raw_se = params[raw_key]['SE']

                accumulative_loss += np.abs(mean - true_mean)
                accumulative_loss_clinical += np.abs(clinical_mean - mean)


                if mean is not None and se is not None:
                    # Format and append to row
                    row[param.upper()] = f"\\makecell{{\\textcolor{{green}}{{{raw_mean:.3f}±{raw_se:.3f}}} \\\\ \\textcolor{{red}}{{{mean:.3f}±{se:.3f}}} \\\\ \\textcolor{{blue}}{{{true_mean:.3f}±{true_se:.3f}}}\\\\ \\textcolor{{black}}{{{clinical_mean:.3f}±{clinical_se:.3f}}}}}"
                else:
                    not_none = False
                    break

            # Append the row to the list
            if not_none: table_rows.append(row)
            else:
                not_none=True


        if print_each_pat:

            df = pd.DataFrame(table_rows).to_latex(index=False)
            df = df.replace(r"\toprule", "").replace(r"\midrule", "").replace(r"\bottomrule","").strip()
            new_lines = []
            lines = df.split("\n")
            for line in lines:
                new_lines.append(line)
                if "&" in line or 'begin' in line:  # This ensures we only add \hline after data rows
                    new_lines.append(r"\hline")

            df = "\n".join(new_lines)

            print('green is raw ADC, red is AI predicted, blue is OBSIDIAN, black is clinical')
            print(f'Average of all runs for {pat_key} with True loss {accumulative_loss:.3f} and Clinical loss {accumulative_loss_clinical:.3f}:\n {df}')

    table_rows = []
    accumulative_loss = 0
    accumulative_loss_clinical = 0

    for region, params in average_all_pat.items():
        row = {'Region': region}
        not_none = True
        for param, values in params.items():
            if 'true' in param or 'clinical' in param or 'raw' in param or 'K_' in param:
                continue

            print(param)
            mean = params[param]['means']
            se = params[param]['SE']

            true_key = [s for s in params.keys() if 'true' in s and not 'K' in s and s.split('_')[1].lower() == param.lower()][0]
            true_mean = params[true_key]['means']
            true_se = params[true_key]['SE']

            clinical_key = [s for s in params.keys() if 'clinical' in s and s.split('_')[1].lower() == param.lower()][0]
            clinical_mean = params[clinical_key]['means']
            clinical_se = params[clinical_key]['SE']

            raw_key = [s for s in params.keys() if 'raw' in s and s.split('_')[1].lower() == param.lower()][0]
            raw_mean = params[raw_key]['means']
            raw_se = params[raw_key]['SE']

            accumulative_loss += np.abs(true_mean - mean)
            accumulative_loss_clinical += np.abs(clinical_mean - mean)
            # Check if both mean and SE are not None
            if mean is not None and se is not None:
                # Format and append to row
                row[param.upper()] = f"\\makecell{{\\textcolor{{green}}{{{raw_mean:.3f}±{raw_se:.3f}}} \\\\  \\textcolor{{red}}{{{mean:.3f}±{se:.3f}}} \\\\ \\textcolor{{blue}}{{{true_mean:.3f}±{true_se:.3f}}}\\\\ \\textcolor{{black}}{{{clinical_mean:.3f}±{clinical_se:.3f}}}}}"
            else:
                # If mean or SE is None, set it as 'N/A' or skip
                not_none = False
                break

        # Append the row to the list
        if not_none:
            table_rows.append(row)
        else:
            not_none = True
    df = pd.DataFrame(table_rows).to_latex(index=False)
    df = df.replace(r"\toprule", "").replace(r"\midrule", "").replace(r"\bottomrule", "").strip()
    new_lines = []
    lines = df.split("\n")
    for line in lines:
        new_lines.append(line)
        if "&" in line or 'begin' in line:  # This ensures we only add \hline after data rows
            new_lines.append(r"\hline")

    df = "\n".join(new_lines)

    print_table = f'True loss: {accumulative_loss:.3f}\\\\Clinical loss: {accumulative_loss_clinical:.3f}\\\\ \n{df}'
    table_ADC_print_toappend[model_name][fitting_name] = f'\\makecell{{{print_table}}}'
    print(f'Average of all patients:\n{print_table}')

    row_string = ''
    fitting_name_dict = {'biexp': 'Biexponential',
                         'kurtosis': 'Kurtosis',
                         'gamma': 'Gamma'}
    model_name_dict = {'attention_unet': 'Attention U-Net',
                       'unet': 'Standard U-Net',
                       'res_atten_unet': 'Residual Attention U-Net',
                       'res_atten_unet_aware': 'Residual Attention U-Net (Noise Aware)',
                       'attention_unet_rician': 'Attention U-Net(Direct)',
                       'unet_rician': 'Standard U-Net(Direct)',
                       'res_atten_unet_rician': 'Residual Attention U-Net(Direct)',
                       'transformer': 'Transformer'}
    oredered_regions = ['Healthy\\_peri','Healthy\\_cent','Lesion\\_peri','Lesion\\_cent']#,'Lesion\\_cent']
    precisionavg = 3
    precisionstd = 3
    significanceThreshold = 0.05
    if first_time == True:
        row_string += 'Clinical                    & Monoexponential                    & '
        for i,r in enumerate(oredered_regions):
            HP_m = average_all_pat[r]['clinical_ADC']['means']
            HP_std = average_all_pat[r]['clinical_ADC']['SE']
            if average_all_pat[r]['clinical_ADC']['p-adc']<=significanceThreshold:
                signif = '${}^{*}$'
            else:
                signif = ''
            if i == len(oredered_regions) -1:row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} \\\\\n'
            else: row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif}  & '
        row_string += 'Raw (No averaging)                   & Monoexponential                    &'
        for i, r in enumerate(oredered_regions):
            HP_m = average_all_pat[r]['raw_ADC']['means']
            HP_std = average_all_pat[r]['raw_ADC']['SE']
            if average_all_pat[r]['raw_ADC']['p-adc'] <=significanceThreshold:
                signif = '${}^{*}$'
            else:
                signif = ''
            if i == len(oredered_regions) -1:row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} \\\\\n'
            else: row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} & '

        row_string += f'True SNR                   & {fitting_name_dict[fitting_name]}                     &'
        for i, r in enumerate(oredered_regions):
            HP_m = average_all_pat[r]['true_SNR']['means']
            HP_std = average_all_pat[r]['true_SNR']['SE']
            if i == len(oredered_regions) - 1:
                row_string += f'{HP_m:.2f} $\\pm$ {HP_std:.2f} \\\\\n'
            else:
                row_string += f'{HP_m:.2f} $\\pm$ {HP_std:.2f} & '

    if new_for_obs:
        row_string += f'OBSIDIAN                   & {fitting_name_dict[fitting_name]}                     & '
        for i, r in enumerate(oredered_regions):

            HP_m = average_all_pat[r]['true_ADC']['means']
            HP_std = average_all_pat[r]['true_ADC']['SE']
            if i == len(oredered_regions) -1:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} \\\\\n'
            else:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} & '

        if fitting_name == 'kurtosis':
            row_string += f'OBSIDIAN (ADC-K)                   & {fitting_name_dict[fitting_name]}                     & '
            for i, r in enumerate(oredered_regions):

                HP_m = average_all_pat[r]['Ktrue_ADC']['means']
                HP_std = average_all_pat[r]['Ktrue_ADC']['SE']
                if i == len(oredered_regions) - 1:
                    row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} \\\\\n'
                else:
                    row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} & '
            row_string += f'OBSIDIAN (K)                   & {fitting_name_dict[fitting_name]}                     & '
            for i, r in enumerate(oredered_regions):

                HP_m = average_all_pat[r]['KKtrue_ADC']['means']
                HP_std = average_all_pat[r]['KKtrue_ADC']['SE']
                if i == len(oredered_regions) - 1:
                    row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} \\\\\n'
                else:
                    row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f} & '

    row_string += f'{model_name_dict[model_name]}                   & {fitting_name_dict[fitting_name]}                     & '
    for i, r in enumerate(oredered_regions):
        HP_m =  average_all_pat[r]['ADC']['means']
        HP_std =average_all_pat[r]['ADC']['SE']
        if average_all_pat[r]['ADC']['p-adc'] <=significanceThreshold:
            signif = '${}^{*}$'
        else:
            signif = ''
        if i == len(oredered_regions)-1:
            row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} \\\\\n'
        else:
            row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} & '
    if fitting_name == 'kurtosis':
        row_string += f'{model_name_dict[model_name]} (ADC-K)                & {fitting_name_dict[fitting_name]}                     & '
        for i, r in enumerate(oredered_regions):
            HP_m = average_all_pat[r]['K_ADC']['means']
            HP_std = average_all_pat[r]['K_ADC']['SE']
            if average_all_pat[r]['K_ADC']['p-adck'] <=significanceThreshold:
                signif = '${}^{*}$'
            else:
                signif = ''
            if i == len(oredered_regions) - 1:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} \\\\\n'
            else:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} & '

        row_string += f'{model_name_dict[model_name]} (K)                & {fitting_name_dict[fitting_name]}                     & '
        for i, r in enumerate(oredered_regions):
            HP_m = average_all_pat[r]['KK_ADC']['means']
            HP_std = average_all_pat[r]['KK_ADC']['SE']
            if average_all_pat[r]['KK_ADC']['p-k'] <=significanceThreshold:
                signif = '${}^{*}$'
            else:
                signif = ''
            if i == len(oredered_regions) - 1:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} \\\\\n'
            else:
                row_string += f'{HP_m:.{precisionavg}f} $\\pm$ {HP_std:.{precisionstd}f}{signif} & '
    table_ADC_print_toappend_flat += row_string

def makeADCTable():
    global dict_runs_pixels_ADC
    global custom_patient

    first_time = True
    #model_name_list = ['unet_rician','unet','attention_unet', 'res_atten_unet']  # 'unet_rician' #add more
    #model_list = [MAIN_METHOD+'_no_rice/unet'] + [MAIN_METHOD+'/'+model for model in ['unet', 'attention_unet', 'res_atten_unet']]
    model_name_list = ['res_atten_unet_aware']# ,'res_atten_unet']  # 'unet_rician' #add more

    model_list = [MAIN_METHOD+'_feed_sigma/res_atten_unet']# + [MAIN_METHOD+'/'+model for model in ['res_atten_unet']]

    fitting_name_list = ['kurtosis']#,'biexp','gamma']#SNR calc based on the first model for OBSIDIAN
    for fitting_name in fitting_name_list:
        new_for_obs = True
        for i,model in enumerate(model_list):
            print('#########################################################\n'         f'Running {model} with {fitting_name} for MAIN_METHOD {MAIN_METHOD}\n'           '#########################################################')
            ADCTable(model_name= model_name_list[i],model = model,fitting_name=fitting_name,MAIN_METHOD = MAIN_METHOD,print_each_pat=True,print_each_run=True, first_time=first_time,custom_patient=custom_patient, standard_error=False, simple_std=True,new_for_obs=new_for_obs)
            first_time = False
            new_for_obs = False

    print(table_ADC_print_toappend_flat)#To print the latex table to be used in the article
    dict_runs_pixels_ADC = to_normal_dict(dict_runs_pixels_ADC)
    import pickle

    save_path = os.path.join(FIGURE_SAVE_PATH,'ADC_pixels.pkl')
    # Save to pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(dict_runs_pixels_ADC, f)
