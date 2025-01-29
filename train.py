import math

from fontTools.misc.psOperators import ps_integer
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from yaml import compose

from model.res_attention_unet import Res_Atten_Unet
from utils import pre_data, post_processing, patientDataset, init_weights
from model.unet_model import UNet
from model.unet_MultiDecoder import UNet_MultiDecoders
from model.attention_unet import Atten_Unet
from model.unet_model import UNet
from pathlib import Path
import logging
import torchvision
import wandb
import argparse
import torch
from IPython import embed
import numpy as np
from pytorch_msssim import MS_SSIM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
import os
dir_checkpoint = Path('../checkpoints/')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # or '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'  # Any available port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ssim_loss = MS_SSIM(channel=20,win_size=5)
        self.mse_loss =  nn.MSELoss()
    def update_data_range(self, range):
        self.ssim_loss.data_range = range
    def forward(self, M,images):
        # Example: Mean squared error + L1 regularization
        loss_ssim = 1 - self.ssim_loss(M, images)
        loss_mse = self.mse_loss(M, images)
        return loss_ssim * loss_mse

def train_net(dataset, net, b, input_sigma: bool,experiment, world_size=None,rank = None,device = None,  epochs: int=5, batch_size: int=2, learning_rate: float = 1e-5,
    val_percent: float=0.1, save_checkpoint: bool=True, sweeping = False):
    b = b.reshape(1, len(b), 1, 1)
    # split into training and validation set
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    if sweeping:
        sampler = None
        print(f'sweeping and using device {device}')
    else:
        print(f'not sweeping and using rank {rank}')

        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)


    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False if not sweeping else True,sampler = sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    if sweeping:
        logging.info(f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {learning_rate}
                Training size:   {n_train}
                Validation size: {n_val}
                Checkpoints:     {save_checkpoint}
                Device/rank:     {device}
        ''')
    else:
        logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device/rank:     {rank}
''')

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    #criterion = nn.MSELoss()
    criterion = CustomLoss()
    global_step = 0
    if rank ==0 or sweeping:
        wandb.watch(models=net, criterion=criterion, log="all", log_freq=10)
    if sweeping:
        rank = device
        world_size=1

    post_process= post_processing()

    for epoch in range(1, epochs+1):
        net.train()
        avg_loss = 0
        num_batches = len(train_loader)

        with tqdm(total=n_train//world_size+1, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                   #(batch,_,_)
            for i, (images,image_b0,sigma,scale_factor) in enumerate(train_loader):

                images = images.to(rank, dtype=torch.float32, non_blocking=True)
                sigma = sigma.to(rank, dtype=torch.float32, non_blocking=True)
                image_b0 = image_b0.to(rank, dtype=torch.float32, non_blocking=True)
                scale_factor = scale_factor.to(rank, dtype=torch.float32, non_blocking=True)
                b = b.to(rank, dtype=torch.float32, non_blocking=True)
                if torch.isnan(images).sum() > 0 or torch.max(images) > 1e10:
                    print(f'-Warning: One batch {i} contained {torch.isnan(images).sum().item()} NaN values and {torch.max(images)} as maximum value.\n This batch was skipped.\n')
                    continue


                if sweeping:
                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                else:
                    assert images.shape[1] == net.module.n_channels, \
                        f'Network has been defined with {net.module.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                M, _, _, _, _ = net(images,b,image_b0, sigma,scale_factor)
                M = M*scale_factor.view(-1,1,1,1)
                images = images*scale_factor.view(-1,1,1,1)
                criterion.update_data_range(torch.max(images))

                loss = criterion(M,images)
                loss.backward()

                # Print the maximum gradient before clipping
                
                max_grad_before = max(p.grad.abs().max().item() for p in net.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.5)
                # Clip gradients to a maximum value
                          
                optimizer.step()
                optimizer.zero_grad()

                #if math.isnan(loss.item()):
                #    print("Error: Loss is NaN")
                #    continue

                global_step += 1
                if rank ==0 or sweeping:
                        experiment.log({
                        'train loss': loss.item(),
                        'max gradient before clipping': max_grad_before,
                        'step': global_step,
                        'epoch': epoch
                    })

                avg_loss += loss.item()

                if rank == 0 or sweeping:
                    pbar.update(images.shape[0])

            if rank == 0 or sweeping:
                with torch.no_grad():
                    val_loss, params, M, img,sig = post_process.evaluate(val_loader, net, rank, b, input_sigma=input_sigma)#sig lonot work
                scheduler.step(val_loss)

                logging.info('Validation Loss: {}'.format(val_loss))
                experiment.log({'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Loss': val_loss,
                            'Max M': M.cpu().max(),
                            'Min M': M.cpu().min(),
                            'Max d1': params['d1'][0].cpu().max(),
                            'Min d1': params['d1'][0].cpu().min(),
                            'Max d2': params['d2'][0].cpu().max(),
                            'Min d2': params['d2'][0].cpu().min(),
                            'Max f': params['f'][0].cpu().max(),
                            'Min f': params['f'][0].cpu().min(),
                            'max Image': img.cpu().max(),
                            'min Image': img.cpu().min(),
                            'd1': wandb.Image(params['d1'][0].cpu()),
                            'd2': wandb.Image(params['d2'][0].cpu()),
                            'f': wandb.Image(params['f'][0].cpu()),
                            'sigma_true' if input_sigma else 'predicted_sigma': wandb.Image(sig.cpu()),
                            'M': wandb.Image(M.cpu()),
                            'image': wandb.Image(img.cpu()),
                            'epoch': epoch,
                            'avg_loss':avg_loss/num_batches
                            })


        # save the model for the current epoch
        if rank==0 and save_checkpoint and not os.getenv("WANDB_SWEEP_ID"):

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        elif sweeping and save_checkpoint and  os.getenv("WANDB_SWEEP_ID") and epoch == epochs:
            sweep_check_path  = Path(os.path.join(dir_checkpoint,'Sweep'))
            sweep_check_path.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(sweep_check_path / f'Batch_size {batch_size} num_epochs {epochs} lr {learning_rate:.4f}_.pth'))
            logging.info(f'Sweep run (Batch_size {batch_size} num_epochs {epochs} lr {learning_rate:.4f}) saved!')
    
    if rank ==0 or sweeping:
        experiment.finish()

    if not sweeping:
        dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=12, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=8e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--patientData', '-dir', type=str, default='/m2_data/mustafa/patientData/', help='Enther the directory saving the patient data')
    parser.add_argument('--diffusion-direction', '-d', type=str, default='M', help='Enter the diffusion direction: M, I, P or S', 
                        dest='dir')
    parser.add_argument('--parallel_training', '-parallel', action='store_true', help='Use argument for parallel training with multiple GPUs.')
    parser.add_argument('--sweep', '-sweep', action='store_true', help='Use this flag if you want to run hyper parameter tuning')
    parser.add_argument('--custom_patient_list', '-clist', type=str, help='Input path to txt file with patient names to be used.')#default='new_patientList.txt'
    parser.add_argument('--input_sigma', '-s', default=True, help='Use argument if sigma map is used as input.')




    return parser.parse_args()


def main(rank,world_size ,sweep):
    if not sweep:
        setup(rank,world_size)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    args = get_args()
    data_dir = args.patientData

    if args.custom_patient_list:
        with open(args.custom_patient_list, 'r') as file:
            # Read the entire file content and split by commas
            content = file.read().strip()  # Remove leading/trailing whitespace (if any)
            patient_list = content.split(',')
        patientData = patientDataset(data_dir,input_sigma=args.input_sigma,  custom_list=patient_list, transform=False)
    else:
        patientData = patientDataset(data_dir,input_sigma=args.input_sigma, transform=False)




    if rank ==0:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    b = torch.linspace(0, 2000, steps=21).cuda(non_blocking=True)
    b = b[1:]

    n_channels = 20
    n_mess = "atten_unet"
    net = Atten_Unet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma).cuda()
    if rank == 0:
        print("Using ", torch.cuda.device_count(), " GPUs!\n")
        logging.info(f'Network:\n'
                     f'\t{n_mess}\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    if args.load:

        if rank == 0: logging.info(f'Model loaded from {args.load}')
        net.load_state_dict(torch.load(args.load))

    net.apply(init_weights)
    if not sweep:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    device = None
    if os.getenv("WANDB_SWEEP_ID"):
        print('running swep')
        experiment = wandb.init()  # Automatically pulls sweep parameters
        config = wandb.config
        wandb.run.name = str(f'Batch_size {config.batch_size} num_epochs {config.epochs} lr {config.learning_rate:.4f}')

        epochs = config['epochs']
        batch_size =  config['batch_size']
        learning_rate =  config['learning_rate']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

    else:
        print('not running swep')

        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.lr
        experiment = None

        if rank==0:
            experiment = wandb.init(project='UNet-Denoise', resume='allow', anonymous='must')
            experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      val_percent=args.val / 100))


    try:
        if sweep:
            train_net(dataset=patientData,
                          net=net,
                          world_size=world_size,
                          b = b,
                          epochs=epochs,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          val_percent=args.val / 100,
                          input_sigma=args.input_sigma,
                      experiment = experiment,
                      save_checkpoint=True,
                      sweeping=True,
                      device=device,
                      )
        else:
            train_net(dataset=patientData,
                      net=net,
                      rank=rank,
                      world_size=world_size,
                      b=b,
                      epochs=epochs,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      val_percent=args.val / 100,
                      input_sigma=args.input_sigma,
                      experiment=experiment,
                      save_checkpoint=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

if __name__ == "__main__":

    if os.getenv("WANDB_SWEEP_ID"):

        try:
            main(rank = 0, world_size= None, sweep = True)
        except KeyboardInterrupt:

            logging.info('Exited')
            raise

    else:
        sweep = False
        world_size = torch.cuda.device_count()  # Number of GPUs
        mp.spawn(main, args=(world_size,sweep), nprocs=world_size)