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
            'values': [0.8, 0.08, 0.008]
        },
        'batch_size': {
            'values': [10,15,20]
        },
        'epochs': {
            'values': [20,30]
        }
    }
}
# Define your sweep ID
sweep_id = wandb.sweep(sweep_config, project="sweepDenoiseMRI")
print(f"Sweep ID: {sweep_id}")