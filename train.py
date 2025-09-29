import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split,Subset
from model.res_attention_unet import Res_Atten_Unet
from utils import post_processing, patientDataset, init_weights
from IPython import embed
from model.attention_unet import Atten_Unet
from model.unet_model import UNet
from pathlib import Path
import logging
import wandb
import argparse
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
import os
import torchvision.models as models
from sklearn.model_selection import KFold

def in_docker():
    # Check for /.dockerenv (standard Docker flag)
    if os.path.exists('/.dockerenv'):
        return True
    # Check cgroup info for Docker or container runtime
    try:
        with open('/proc/1/cgroup', 'rt') as f: return any('docker' in line or 'kubepods' in line or 'containerd' in line for line in f)
    except Exception:
        return False


#Directory for net models to be saved at as .pth files
dir_checkpoint = Path('/write/your/path/checkpoints') if not in_docker() else Path('/app/checkpoints')
os.environ["WANDB_BASE_URL"] = "http://localhost:8080" if not in_docker() else "http://wandb:8080"
dataset_path1 = '/write/your/path'  if not in_docker() else '/host-trainset1/'
dataset_path2 = '/write/your/path' if not in_docker() else '/host-trainset2/'
early_termination = False

def setup(rank, world_size):
    """
    This function assigns a **master machine** and **port** used for multiprocessing used by *torch.nn.parallel.DistributedDataParallel*.
    The GPUs will communicate thorough the assigned port.\n
    Standard port is 12355.

    :param rank: Unique GPU-ID passed as an integer. Ranges from 0 to N-1 if machine has N GPUs
    :type rank: int or string


    :param world_size: Total number of processes (or GPUs) used for parallel training, e.g if 2 machines are used, with 4 GPUs each, then world_size = 8
    :type world_size: int or string
    """
    os.environ['MASTER_ADDR'] = 'localhost'# If a remote computer is used as master machine, assign the machine's IP, e.g '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'# Any available port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    #nccl is a type of communication backend for GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)#nccl
    torch.cuda.set_device(rank)



class CustomLoss(nn.Module):

    """
    Custom loss function: :math:`L` = (1-SSIM) :math:`\cdot` L1

    Example:

        >>>loss = CustomLoss()

        >>>loss_value = loss(predicted,target)
    """

    def __init__(self):
        super(CustomLoss, self).__init__()
        # For one dimensional channel, for example when inputting ADC map in loss function.

        self.l1_loss =  nn.L1Loss()#nn.MSELoss()

    def forward(self, M,images, ssim_bool = False, only_ssim = False):
        """
            :param M: The predicted data/image
            :param images: The target data/image

            """
        l1_loss = self.l1_loss(M,images)

        return l1_loss

def train_net(train_set,val_set,run_number, net, b, input_sigma: bool,experiment, training_model: str, fitting_model: str,
              world_size=None,rank = None,device = None,  epochs: int=30, batch_size: int=1, learning_rate: float = 1e-3,
              save_checkpoint: bool=True, sweeping = False):

    """
    Main function used for training *net*. This function is multi functional and can run on single GPU and on multiple GPUs if available.
    It can also run hyperparameter tuning with wand.sweep.

    :param dataset: Dataset as type *torch.utils.data.dataset*.
    :type dataset: torch.utils.data.Dataset

    :param net: Network model as type *torch.nn.Module*.
    :type net: torch.nn.Module

    :param b: Diffusion weighting (b-values) as type *torch.tensor*, e.g [100.,200.,...,2000.]. Dimension must match with data from your dataset and network structure.
    :type b: torch.Tensor

    :param input_sigma: Pass True if noise map is inputted to network. If passed False then network will use its own generated noise map that is learned in an unsupervised manner.

    :param experiment: A wandb.run object returned by init used for logging: >>>experiment = wandb.init().

    :param training_model: Name of network model, possible values 'unet'/'attention_unet'/'res_atten_unet' for UNet/Attention UNet/Residual Attention UNet.

    :param fitting_model: Name of fitting model, possible values 'biexp'/'kurtosis'/'gamma'.

    :param world_size: Total number of processes (or GPUs) used for parallel training, e.g if 2 machines are used, with 4 GPUs each, then world_size = 8.

    :param rank: Unique GPU-ID passed as an integer. Ranges from 0 to N-1 if machine has N GPUs.

    :param device: Type of computation device ('cpu' or 'cuda:0'). **Only** input device when running one GPU.
    :type device: torch.device

    :param epochs: Number of epochs to train, *default*: 30.

    :param batch_size: Size of each batch from dataset, *default*: 1.

    :param learning_rate: Learning rate, *default*: 1e-3.

    :param val_percent: Percentage of all data to be used as validation data. Values between 0-1,  *default*: 0.1.

    :param save_checkpoint: True if model weights are to be saved during/after training

    :param sweeping: True if hyperparameter tuning is being performed. The script/function train_net() will run separately on all GPUs

    :return: None
    """

    args = get_args()#Getting arguments passed from CLI through ArgumentParser.

    assert not(args.feed_sigma and not args.input_sigma), 'Error: Argument input_sigma needs to be true if argument feed_sigma is passed'
    assert not(not args.estimate_sigma and not args.input_sigma), 'Error: Both estimate_sigma and input_sigma are false, one of them must be true since a noise map is needed to function.'

    b = b.reshape(1, len(b), 1, 1)#Reshaped to match dimension of data (num_slices, num_diffusion_levels, width, height)

    if sweeping:
        #During a sweep (hyperparameter tuning) each wandb.agent is assigned one GPU (different processes/network are trained in parallel on different GPUs).
        #Each GPU will need the whole dataset, as they don't share networks.
        #Thus, no sampling/distribution of data will be done between GPUs as done in parallel training for one network.
        sampler = None
        print(f'Sweeping and using device {device}')
    else:
        #Since we are training one network, it can be trained in parallel with multiple GPUs.
        #Data can be partitioned and distributed to each GPU
        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)#num_replicas: Number of partitions
        print(f'Not sweeping and using rank {rank}')


    #For large data, num_workers must be low, as each worker will have a copy of the whole data to RAM.
    train_loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader_args = dict(batch_size=min(10,len(val_set)//2), num_workers=0, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=False if not sweeping else True,sampler = sampler, **train_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)


    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {len(train_set)}
            Validation size: {len(val_set)}
            Checkpoints:     {save_checkpoint}
            Device/rank:     {device if sweeping else rank}
    ''')    #Ranks only relevant in parallel training. In sweep-mode, each GPU has a separate training instance


    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2 ,mode = 'min', patience=8, min_lr = 6e-07)#3 normal, 8 with 200 ep no termination
    #If validation loss does not increase in two consecutive epochs, lr decreases


    criterion = CustomLoss()
    global_step = 0
    if rank ==0 or sweeping:
        #Used for logging weights and gradients
        #One network on multiple GPUs: To avoid double logging same network, one GPU with ID (rank=0) logs all.
        #Multiple networks, each on one GPU (sweep): All networks are logged by their respective GPU, hence the sweeping.

        wandb.watch(models=net, criterion=criterion, log="all", log_freq=10)
    if sweeping:
        rank = device#GPU-ID: torch.device used during tensor.to().
        world_size=1#Training session is run by one GPU

    post_process= post_processing()#Module used for validation of network during training
    overfitting_patience = 5  # Stop if no improvement after 5 epochs
    overfitting_counter = 0
    last_avg_loss = None
    best_val_loss = None

    for epoch in range(1, epochs+1):
        net.train()
        avg_loss = 0
        num_batches = len(train_loader)

        with tqdm(total=len(train_set)//world_size+1, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for i, (images,image_b0,sigma,scale_factor) in enumerate(train_loader):
                #scale_factor: Scaling used on sample images in dataset before forward pass

                # non_blocking=True for asynchronous data transfer (happens in background and simultaneously with other tasks (e.g. computation))
                images = images.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 20 or 60 if use_3D , 200, 240)
                sigma = sigma.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 1, 200, 240)
                image_b0 = image_b0.to(rank, dtype=torch.float32, non_blocking=True)# (n_batches, 1, 200, 240)
                scale_factor = scale_factor.to(rank, dtype=torch.float32, non_blocking=True)#(n_batches,)
                b = b.to(rank, dtype=torch.float32, non_blocking=True)#(1, 20, 1 , 1)

                if torch.isnan(images).sum() > 0 or torch.max(images) > 1e10:
                    print(f'-Warning: One batch {i} contained {torch.isnan(images).sum().item()} NaN values and {torch.max(images)} as maximum value.\n This batch was skipped.\n')
                    continue


                if sweeping:
                    #If number of b-values does not match with number of input channel to net
                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                else:
                    assert images.shape[1] == net.module.n_channels, \
                        f'Network has been defined with {net.module.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                #M has same shape as images (num_batches,num_diffusion_levels, width, height)
                M, outdict = net(images,b,image_b0, sigma,scale_factor)#returnes tuple (M:output_image, dictionary_of_predicted_parameter_values)
                # M: (n_batches, 20 or 60 if use_3D, 200, 240)
                # param_dict: dict_keys(['parameters', 'sigma', 'names']),  'names' for parameter names

                #Rescale output and input images, as they were normalized in dataset.
                M = M*scale_factor.view(-1,1,1,1)
                images = images*scale_factor.view(-1,1,1,1)
                sigma = sigma*scale_factor.view(-1,1,1,1)
                if 'estimated_sigma' in outdict:
                    predicted_sigma = outdict['estimated_sigma']
                    predicted_sigma = predicted_sigma



                loss = criterion(M,images)
                if 'estimated_sigma' in outdict and args.include_sigma_loss:
                    sigma_loss = criterion(predicted_sigma, sigma)
                else:
                    sigma_loss = 0
                loss += sigma_loss

                loss.backward()


                #Maximum gradient before clipping. For logging
                max_grad_before = max(p.grad.abs().max().item() for p in net.parameters() if p.grad is not None)

                #Clip gradients to a maximum value
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                if rank ==0 or sweeping:
                    #Log by one GPU
                    experiment.log({
                        'train loss': loss.item(),
                        'sigma loss': sigma_loss.item() if not isinstance(sigma_loss, (int, float)) else sigma_loss,
                        'max gradient before clipping': max_grad_before,
                        'step': global_step,
                        'epoch': epoch
                    })

                avg_loss += loss.item()

                if rank == 0 or sweeping:
                    pbar.update(images.shape[0])

            avg_loss = avg_loss/num_batches


            if sweeping and epoch > 1 and avg_loss>last_avg_loss+10:
                save_path = Path(
                    os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model, f'run_{run_number}'))
                Path(save_path).mkdir(parents=True, exist_ok=True)
                if os.path.exists(str(save_path / 'checkpoint_best.pth')):
                    net.load_state_dict(torch.load(str(save_path / 'checkpoint_best.pth'), weights_only=True))
                    logging.info(f'Spike in loss detected ({last_avg_loss:.3f} to {avg_loss:.3f}), loading best checkpoint.')
                else:
                    logging.info(f'Spike in loss detected ({last_avg_loss:.3f} to {avg_loss:.3f}), but no best checkpoint found.')
                continue
            last_avg_loss = avg_loss


            with torch.no_grad():
                val_loss, params, save_dict, M, img,sig = post_process.evaluate(val_loader, net, rank, b, input_sigma=input_sigma,estimate_sigma=args.estimate_sigma, use_3D=args.use_3D, include_sigma_loss=args.include_sigma_loss)

            if epoch == 1:
                best_val_loss = val_loss

            if (rank == 0 or sweeping) and val_loss<best_val_loss:

                best_val_loss = val_loss
                save_path = Path(os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model, f'run_{run_number}'))
                Path(save_path).mkdir(parents=True, exist_ok=True)
                if epoch <=200:
                    torch.save(net.state_dict(), str(save_path / 'checkpoint_best200.pth'))
                else:
                    torch.save(net.state_dict(), str(save_path / 'checkpoint_best.pth'))
                logging.info(f'Checkpoint {epoch} saved as best!')



            scheduler.step(torch.round(val_loss*1000)/1000)
            # The mul. with 100 and rounding is a workaround to have
            # scheduler only look at 2 decimals
            if rank == 0 or sweeping:

                logging.info('Validation Loss: {}'.format(val_loss))
                logging_dict = {'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Loss': val_loss,
                            'Max M': M.cpu().max(),
                            'Min M': M.cpu().min(),
                            'max Image': img.cpu().max(),
                            'min Image': img.cpu().min(),
                            'sigma_scale': net.sigma_scale.item() if os.getenv("WANDB_SWEEP_ID") else net.module.sigma_scale.item(),
                            'epoch': epoch,
                            'avg_loss':avg_loss
                            }

                logging_dict.update(params)#log model parameters
                save_dict.update({
                    'M': M.cpu(),
                    'image': img.cpu()
                })
                image_save_path = Path(os.path.join(experiment.dir,'images'))
                image_save_path.mkdir(parents=True, exist_ok=True)
                for k,v in save_dict.items():#Locally save images

                    if isinstance(v, np.ndarray):
                        res = v
                    else:
                        res = v.cpu().numpy()


                    res_min = np.min(res)

                    res_max = np.max(res)

                    logging_dict.update({f'{k}': wandb.Image(res / res_max),

                                     f'{k}_min': res_min,

                                     f'{k}_max': res_max,

                                     })
                experiment.log(logging_dict)


            if val_loss < avg_loss+2:# +2 leeway in considering as not overfitting
                if rank == 0 or sweeping:
                    print(f'Not overfitting {rank}: val {val_loss:.3f} and avg_loss {avg_loss:.3f}')
                overfitting_counter = 0  # Reset counter if validation loss improves
            elif epoch>=15 and early_termination:#Overfitting is considered only for epochs>15.
                overfitting_counter += 1
                if rank == 0 or sweeping:
                    print(f'Overfitted: val {val_loss:.4f} and avg_loss {avg_loss:.4f} counter {overfitting_counter}')
                if overfitting_counter >= overfitting_patience:
                    print("Early stopping triggered. Reporting run as finished to wandb.")
                    if rank == 0 or sweeping:
                        if save_checkpoint:
                            save_path = Path(
                                os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model,
                                             f'run_{run_number}'))
                            Path(save_path).mkdir(parents=True, exist_ok=True)
                            torch.save(net.state_dict(), str(save_path / 'checkpoint_epoch{}.pth'.format(epoch)))
                            logging.info(f'Checkpoint {epoch} saved!')
                        wandb.finish(exit_code=0)
                    print("Training stopped early due to overfitting.")
                    break





        # save the model for the current epoch
        if (epoch>epochs-1 or (optimizer.param_groups[0]['lr'] < 3e-5 and early_termination)):#'lr' is too low to learn effectively
            if optimizer.param_groups[0]['lr'] < 3e-5 and early_termination:
                print('Training stopped due to LR < 3e-5')
            if (rank == 0 or sweeping) and save_checkpoint:
                save_path = Path(os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model, f'run_{run_number}'))
                Path(save_path).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(save_path / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                wandb.finish(exit_code=0)
            break
        if epoch == 200:
            if (rank == 0 or sweeping) and save_checkpoint:
                save_path = Path(os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model, f'run_{run_number}'))
                Path(save_path).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(save_path / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    if rank ==0 or sweeping:
        experiment.finish()


    if not sweeping:
        dist.destroy_process_group()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=12, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=8e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--patientData', '-dir', type=str, default=f'{dataset_path1},{dataset_path2}', help='Enter the directory saving the patient data')
    parser.add_argument('--custom_patient_list', '-clist', type=str, help='Input path to txt file with patient names to be used.')#default='new_patientList.txt'
    parser.add_argument('--input_sigma', '-s',  type=str, help='Use argument if sigma map is used as input.')
    parser.add_argument('--training_model', '-trn', default='attention_unet',help='Specify which training model to use. Choose between ...,...,...')
    parser.add_argument('--fitting_model', '-fit', default='biexp', help='Specify which fitting model to use')
    parser.add_argument('--Kfolds', '-kf', default='1', help='Used for K-Fold cross validation, pass a number')
    parser.add_argument('--run_number', '-rn', default='1', help='Pass a run number, ignored if k-fold is used')
    parser.add_argument('--main_folder', '-folder', default='cross_validation_l1', help='Specify main folder name')
    parser.add_argument('--use_3D', '-3d', type= str, help='Pass True if using 3D diffusion')
    parser.add_argument('--learn_sigma_scaling', '-ss', type= str, help='Pass True if allowing for AI to learn scaling sigma')
    parser.add_argument('--estimate_S0', '-ests0', type= str, help='Pass True if allowing for AI to estimate S0-image')
    parser.add_argument('--estimate_sigma', '-estsigma', type= str, help='Pass True if allowing for AI to estimate noise map')
    parser.add_argument('--feed_sigma', '-fs', type= str, help='Pass True if feeding sigma map to AI. Input sigma has to be true')
    parser.add_argument('--include_sigma_loss', '-sigma_loss', type=str,help='Pass True if using noise map inside a loss function')
    parser.add_argument('--use_true_sigma', '-use_true_sigma', type=str,help='Pass True if using noise map from OBSIDIAN in rice calculation')
    parser.add_argument('--run_index', '-run_index', type=str,help='Used by sweep_train.py')
    parser.add_argument('--kfold_seed', '-kfold_seed', type=str,help='Used by sweep_train.py')

    return parser.parse_args()
def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    total_bytes = num_params * 4  # assuming float32 (4 bytes)
    size_MB = total_bytes / (1024 ** 2)
    size_GB = total_bytes / (1024 ** 3)
    print(f"Model has {num_params:,} parameters")
    print(f"Size: {size_MB:.2f} MB ({size_GB:.4f} GB)")

def main(rank,world_size ,sweep, kfold_seed,run_index):

    if not sweep:
        #Setup parallel training on multiple GPUs
        setup(rank,world_size)
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
    args = get_args()
    data_dir = args.patientData
    if args.training_model == 'unetr': model_unetr= True#Required special dimensions for input data (208,240) or (240,240)
    else: model_unetr = False
    #If a select number of patients are used for training and not all patients in data_dir
    if args.custom_patient_list:
        with open(args.custom_patient_list, 'r') as file:
            # Read the entire file content and split by commas
            content = file.read().strip()  # Remove leading/trailing whitespace (if any)
            patient_list = content.split(',')
        #Dataset containing patients from the custom list only
        patientData = patientDataset(data_dir=data_dir,input_sigma=args.input_sigma,  custom_list=patient_list, transform=False, crop = True,model_unetr =  model_unetr, use_3D=args.use_3D, fitting_model = args.fitting_model)
    else:
        #Dataset containing all patients in data_dir
        patientData = patientDataset(data_dir=data_dir,input_sigma=args.input_sigma, transform=False, crop = True,model_unetr =  model_unetr, use_3D=args.use_3D, fitting_model = args.fitting_model)
    if rank ==0:
        #Log by one GPU (with ID = 0) only
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        print(f'Loaded {len(patientData.file_paths)} patients:\n')
        for file in patientData.file_paths: print(file)


    val_percent = args.val / 100

    # split into training and validation set
    n_val = int(len(patientData) * val_percent)
    n_train = len(patientData) - n_val

    n_kfolds = int(args.Kfolds)
    if n_kfolds > 1:
        kfold = KFold(n_splits=n_kfolds,random_state=kfold_seed, shuffle=True)
        _, (train_ids, val_ids) = list(enumerate(kfold.split(patientData)))[run_index]
        train_set = Subset(patientData, train_ids)
        val_set = Subset(patientData, val_ids)
        run_number = run_index+1
        print(f'{val_ids}\n{train_ids}')
    else:
        train_set, val_set = random_split(patientData, [n_train, n_val])
        run_number = int(args.run_number) if args.run_number else 1



    b = torch.linspace(0, 2000, steps=21).cuda(non_blocking=True)
    b = b[1:]
    n_channels = 20
    if args.use_3D: n_channels *= 3


    if args.training_model == 'attention_unet':
        n_mess = "atten_unet"
        net = Atten_Unet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma).cuda()
    elif args.training_model == 'unet':
        n_mess = "unet"
        net = UNet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma).cuda()
    elif args.training_model == 'res_atten_unet':
        n_mess = "res_atten_unet"
        net = Res_Atten_Unet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, estimate_sigma=args.estimate_sigma,use_true_sigma=args.use_true_sigma, feed_sigma=args.feed_sigma).cuda()

    print(get_model_size(net))
    if rank == 0:
        print("Using ", torch.cuda.device_count(), " GPUs!\n")
        logging.info(f'Network:\n'
                     f'\t{n_mess}\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if not sweep:
        # During a sweep (hyperparameter tuning) each wandb.agent is assigned one GPU (different processes/network are trained in parallel on different GPUs).
        # Each GPU will need the whole dataset, as they don't share networks.
        # Thus, no sampling/distribution of data will be done between GPUs as done in parallel training for one network.

        # During a normal training (no sweep), every GPU trains the same network and can be trained in parallel
        # The dataset can be partitioned and distributed to each GPU, were each GPU processes their assigned data through the forward pass of network
        # Weights on each GPU is synchronised and resulted gradient calculations are shared between GPUs.
        # Hence, DistributedDataParallel is used to achieve this and train in parallel
        net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    if args.load:

        if rank == 0: logging.info(f'Model loaded from {args.load}')
        net.load_state_dict(torch.load(args.load))
    else:
        net.apply(init_weights)#Weights initialization


    device = None

    if os.getenv("WANDB_SWEEP_ID"):#Variable exists if sweep is used
        print('Running sweep')
        experiment = wandb.init(reinit=True)
        config = wandb.config# Automatically pulls sweep parameters from configurations of the sweep
        wandb.run.name = str(f'{config.training_model}_{config.fitting_model}_{run_number}')

        epochs = config['epochs']
        batch_size =  config['batch_size']
        learning_rate =  config['learning_rate']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

    else:
        print('not running sweep')

        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.lr
        experiment = None

        if rank==0:
            experiment = wandb.init(project='UNet-Denoise', resume='allow')
            experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      val_percent=args.val / 100))


    try:

        train_net(train_set=train_set,
                  val_set=val_set,
                  run_number  =run_number,
                  net=net,
                  rank = rank if not sweep else None,
                  world_size=world_size,
                  b = b,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  input_sigma=args.input_sigma,
                  experiment = experiment,
                  save_checkpoint=True,
                  sweeping=sweep,#####
                  device=device if sweep else None,#####
                  training_model = args.training_model,
                  fitting_model = args.fitting_model
                  )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

if __name__ == "__main__":

    args = get_args()
    n_kfolds = int(args.Kfolds)
    if args.kfold_seed:
        kfold_seed = int(args.kfold_seed)
    else:
        kfold_seed = random.randint(0, 2 ** 16 - 1)

    if n_kfolds>1:print(f'K-fold seed is {kfold_seed}' )
    if not args.run_index:
        print('No run index')
        for run_index in range(n_kfolds):
            if n_kfolds>1: print(f'Running: Fold {run_index+1}')

            if os.getenv("WANDB_SWEEP_ID"):

                try:
                    main(rank = 0, world_size= None, sweep = True,kfold_seed=kfold_seed,run_index=run_index)
                except KeyboardInterrupt:

                    logging.info('Exited')
                    raise

            else:
                sweep = False
                world_size = torch.cuda.device_count()  # Number of GPUs
                mp.spawn(main, args=(world_size,sweep,kfold_seed,run_index), nprocs=world_size)
    else:
        print(f'We have run index {int(args.run_index)}')
        if os.getenv("WANDB_SWEEP_ID"):

            try:
                main(rank=0, world_size=None, sweep=True, kfold_seed=kfold_seed, run_index=int(args.run_index))
            except KeyboardInterrupt:

                logging.info('Exited')
                raise
