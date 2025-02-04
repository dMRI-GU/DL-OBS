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
            'values': ['1','2','3','4','5']
        },
        'training_model':{
            'values': ['attention_unet', 'unet', 'res_atten_unet']
        },
        'fitting_model': {
            'values': ['biexp','kurtosis', 'gamma']
        },



    }
}
# Define your sweep ID
sweep_id = wandb.sweep(sweep_config, project="sweepDenoiseMRI")
print(f"Sweep ID: {sweep_id}")

#CUDA_VISIBLE_DEVICES=0 wandb agent mustafa-abbas-sahlgrenska-universitetssjukhuset/sweepDenoiseMRI/6lkhhnig
#CUDA_VISIBLE_DEVICES=1 wandb agent mustafa-abbas-sahlgrenska-universitetssjukhuset/sweepDenoiseMRI/6lkhhnig
