import subprocess
from random import random

import wandb
import os
sweep_config = {
    'name': 'sweepDenoiseMRI',
    'program': 'train.py',
    'method': 'grid',
    'metric': {
        'name': 'avg_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [10]
        },
        'epochs': {
            'values': [4]
        },
        'learning_rate': {
            'values': [0.0008]
        },
        'Kfolds':{
            'values':['5']
        },
        'run_index': {
            'values': ['0','1','2','3','4']#All five folds
        },
        'kfold_seed': {
            'values': ['65651']
        },
        'training_model':{
            'values': ['attention_unet','unet','res_atten_unet']
        },
        'fitting_model': {
            'values': ['biexp']
        },
        'main_folder': {
            'values': ['scanner_noise_l1_s0est_3D_test']
        },
        'estimate_S0': {
            'values': ['True']
        },
        #'estimate_sigma': {
        #    'values': ['True']
        #},
        'use_true_sigma': {
            'values': ['True']
        },
        'input_sigma': {
            'values': ['True']
        },
        'use_3D': {
            'values': ['True']
        },

        #'include_sigma_loss':{#Guided approach
        #    'values': ['True']
        # },
        'feed_sigma': {
            'values': ['True']
        },

        #'custom_patient_list':{
        #    'values': ['trainList.txt']
        #},
        #'learn_sigma_scaling': {
        #    'values': ['True']
        #},

    }
}
# Define your sweep ID
sweep_id = wandb.sweep(sweep_config, project="sweepDenoiseMRI")
print(f"Sweep ID: {sweep_id}")

#Example code to execute
print(f'CUDA_VISIBLE_DEVICES=0 wandb agent WandbUserName/sweepDenoiseMRI/{sweep_id}')
print(f'CUDA_VISIBLE_DEVICES=1 wandb agent WandbUserName/sweepDenoiseMRI/{sweep_id}')
