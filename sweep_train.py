import subprocess
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
        'learning_rate': {
            'values': [0.0008]
        },
        'batch_size': {
            'values': [30]
        },
        'epochs': {
            'values': [30]
        },
        'run_number':{
            'values':['1','2','3','4','5'] #['1']#,
        },
        'training_model':{
            'values': ['unet', 'res_atten_unet','attention_unet']
        },
        'fitting_model': {
            'values': ['biexp', 'gamma','kurtosis']#,['biexp'
        },
        #'adc_as_loss':{
        #    'values': ['True']
        #},
        'main_folder': {
            'values': ['corss_validation_l1_ssim_s0est_feed_sigma']
        },
        'estimate_S0': {
            'values': ['True']
        },
        'input_sigma': {
            'values': ['True']
        },
        'feed_sigma': {
            'values': ['True']
        },
        #'use_3D': {
        #    'values': ['True']
        #},
        #'custom_patient_list':{
        #    'values': ['predictList.txt']
        #}
        #'learn_sigma_scaling': {
        #    'values': ['True']
        #},
    }
}
# Define your sweep ID
sweep_id = wandb.sweep(sweep_config, project="sweepDenoiseMRI")
print(f"Sweep ID: {sweep_id}")

#Example code to execute
print(f'CUDA_VISIBLE_DEVICES=0 wandb agent mustafa-abbas-sahlgrenska-universitetssjukhuset/sweepDenoiseMRI/{sweep_id}')
print(f'CUDA_VISIBLE_DEVICES=1 wandb agent mustafa-abbas-sahlgrenska-universitetssjukhuset/sweepDenoiseMRI/{sweep_id}')
