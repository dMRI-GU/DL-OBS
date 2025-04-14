from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from model.res_attention_unet import Res_Atten_Unet
from utils import post_processing, patientDataset, init_weights
from model.unet_MultiDecoder import UNet_MultiDecoders
from model.UNETR import UNETR
from IPython import embed
from model.attention_unet import Atten_Unet
from model.unet_model import UNet
from model.unet_2Decoder import UNet_2Decoders
from pathlib import Path
import logging
import wandb
import argparse
import torch
import numpy as np
from pytorch_msssim import MS_SSIM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
import os
import torchvision.models as models

#Directory for net models to be saved at as .pth files
dir_checkpoint = Path('/TANK/mustafa/checkpoints')


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
    Custom loss function: :math:`L` = (1-SSIM) :math:`\cdot` MSEloss

    Example:

        >>>loss = CustomLoss()

        >>>loss_value = loss(predicted,target)
    """


    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ssim_loss = MS_SSIM(channel=1, win_size=5)
        self.ssim_loss2 = MS_SSIM(channel=20, win_size=5)
        self.ssim_loss3 = MS_SSIM(channel=60, win_size=5)
        self.mse_loss =  nn.L1Loss()#nn.MSELoss()
    def update_data_range(self, range):
        self.ssim_loss.data_range = range
    def forward(self, M,images, ssim_bool = False, only_ssim = False):
        if M.shape[1]>1 and M.shape[1]<21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss2(M, images)
        elif M.shape[1]>21 and ssim_bool:
            loss_ssim = 1 - self.ssim_loss3(M, images)
        elif ssim_bool:
            loss_ssim = 1 - self.ssim_loss(M, images)
        else:
            loss_ssim = 1

        if not only_ssim:

            loss_mse = self.mse_loss(M,images)
        else:
            loss_mse = 1
        return loss_ssim * loss_mse

def train_net(dataset, net, b, input_sigma: bool,experiment, training_model: str, fitting_model: str,run_number: str, world_size=None,rank = None,device = None,  epochs: int=30, batch_size: int=1, learning_rate: float = 1e-3,
    val_percent: float=0.1, save_checkpoint: bool=True, sweeping = False):

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

    :param run_number: A string of single number used for tracking run-ID used for cross validation purposes.

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

    ADC_loss = args.adc_as_loss
    b = b.reshape(1, len(b), 1, 1)#Reshaped to match dimension of data (num_slices, num_diffusion_levels, width, height)

    # split into training and validation set
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

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


    #Num_workers must = 0, as num_workers > 0 duplicates the data to RAM. Bug?
    train_loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=False if not sweeping else True,sampler = sampler, **train_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    criterion = CustomLoss()
    global_step = 0
    if rank ==0 or sweeping:
        #Used for logging weights and gradients
        #One network on multiple GPUs: To avoid double logging same network, one GPU with ID (rank=0) logs all.
        #Multiple networks, each on one GPU (sweep): All networks are logged by their respective GPU, hence the sweeping.

        wandb.watch(models=net, criterion=criterion, log="all", log_freq=10)
    if sweeping:
        rank = device#GPU-ID = torch.device used during tensor.to()
        world_size=1#Function runs by one GPU

    post_process= post_processing()#Module used for validation of network during training
    overfitting_patience = 5  # Stop if no improvement after '5' epochs
    overfitting_counter = 0

    for epoch in range(1, epochs+1):
        net.train()
        avg_loss = 0
        num_batches = len(train_loader)

        with tqdm(total=n_train//world_size+1, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for i, (images,image_b0,sigma,scale_factor) in enumerate(train_loader):
                #images (num_batches,num_diffusion_levels, width, height): Used as target
                #image_b0 (num_batches, width, height)
                #sigma (num_batches, width, height)
                #scale_factor (num_batches,): Scaling used on sample images in dataset before forward pass

                # non_blocking=True for asynchronous data transfer (happens in background and simultaneously with other tasks (e.g. computation))
                images = images.to(rank, dtype=torch.float32, non_blocking=True)
                sigma = sigma.to(rank, dtype=torch.float32, non_blocking=True)
                image_b0 = image_b0.to(rank, dtype=torch.float32, non_blocking=True)
                scale_factor = scale_factor.to(rank, dtype=torch.float32, non_blocking=True)
                b = b.to(rank, dtype=torch.float32, non_blocking=True)

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


                M, _ = net(images,b,image_b0, sigma,scale_factor)#returnes tuple (M:output_image, dictionary_of_predicted_parameter_values)

                #Rescale output and input images, as they were normalized in dataset.

                M = M*scale_factor.view(-1,1,1,1)
                images = images*scale_factor.view(-1,1,1,1)

                if ADC_loss:
                    if args.use_3D: slicing = [[slice(0,1),slice(19,20)],
                                               [slice(20,21),slice(39,40)],
                                               [slice(40,41),slice(59,60)]]
                    else:  slicing = [[slice(0,1),slice(19,20)]]
                    ADC_avg_M = torch.zeros(size = (len(slicing),M.shape[0],1,*M.shape[-2:]),device=rank)
                    ADC_avg_images = torch.zeros(size = (len(slicing),images.shape[0],1,*images.shape[-2:]),device=rank)

                    for diff_index,sl in enumerate(slicing):
                        im100 = images[:,sl[0]]
                        im1000 = images[:,sl[1]]
                        im100= im100+ torch.tensor(0.0001, device=im100.device)
                        im1000= im1000+ torch.tensor(0.0001, device=im100.device)
                        ADC_avg_images[diff_index] = -torch.log(im1000 / im100) / (2000 -100)

                        M100 = M[:, sl[0]]
                        M1000 = M[:,sl[1]]
                        M100= M100+ torch.tensor(0.0001, device=im100.device)
                        M1000= M1000+  torch.tensor(0.0001, device=im100.device)
                        ADC_avg_M[diff_index] = -torch.log(M1000 / M100) / (2000 - 100)
                    ADC_avg_images = torch.mean(ADC_avg_images,dim=0)
                    ADC_avg_M = torch.mean(ADC_avg_M, dim=0)
                ADC_loss_val = 1


                if ADC_loss:
                    criterion.update_data_range(torch.max(ADC_avg_images))
                    loss = 9*1000*criterion(ADC_avg_M, ADC_avg_images, ssim_bool=False)#12, 6
                    ADC_loss_val = loss.item()
                    criterion.update_data_range(torch.max(images))
                    loss += criterion(M, images,  ssim_bool=True)


                    # criterion.update_data_range(torch.max(images[:, 0:1]))
                    # loss *= criterion(M[:, 0:1], images[:, 0:1], ssim_bool=True, only_ssim=True)

                else:
                    criterion.update_data_range(torch.max(images))
                    loss = criterion(M,images, ssim_bool = True)
                loss.backward()

                #Maximum gradient before clipping
                max_grad_before = max(p.grad.abs().max().item() for p in net.parameters() if p.grad is not None)

                #Clip gradients to a maximum value
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1)

                optimizer.step()
                optimizer.zero_grad()

                #if math.isnan(loss.item()):
                #    print("Error: Loss is NaN")
                #    continue

                global_step += 1
                if rank ==0 or sweeping:
                    #Log by one GPU
                    experiment.log({
                        'train loss': loss.item(),
                        'ADC_loss': ADC_loss_val,
                        'max gradient before clipping': max_grad_before,
                        'step': global_step,
                        'epoch': epoch
                    })

                avg_loss += loss.item()

                if rank == 0 or sweeping:
                    pbar.update(images.shape[0])

            with torch.no_grad():
                val_loss, params, save_dict, M, img,sig = post_process.evaluate(val_loader, net, rank, b, input_sigma=input_sigma, ADC_loss= ADC_loss, use_3D=args.use_3D)
            scheduler.step(torch.round(val_loss*10000)/10000)
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
                            'avg_loss':avg_loss/num_batches
                            }

                logging_dict.update(params)#log model parameters
                save_dict.update({
                    'sigma_true' if input_sigma else 'predicted_sigma':sig.cpu(),
                    'M': M.cpu(),
                    'image': img.cpu()
                })
                image_save_path = Path(os.path.join(experiment.dir,'images'))
                image_save_path.mkdir(parents=True, exist_ok=True)
                experiment.log(logging_dict)
                for k,v in save_dict.items():
                    plt.imsave(os.path.join(image_save_path,f"{k}_{global_step}.png"), v, cmap="gray")  # Use cmap="gray" for grayscale images
            if val_loss < avg_loss/num_batches+2:# not overfitting
                if rank == 0 or sweeping:
                    print(f'not over: val {val_loss:.4f} and avg_loss {avg_loss/num_batches:.4f}')
                overfitting_counter = 0  # Reset counter if validation loss improves
            elif epoch>=15:
                overfitting_counter += 1
                if rank == 0 or sweeping:
                    print(f'overfitted: val {val_loss:.4f} and avg_loss {avg_loss/num_batches:.4f} counter {overfitting_counter}')
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
        if (epoch>29 or optimizer.param_groups[0]['lr'] < 1e-6):#save every 5 epoch

            if (rank == 0 or sweeping) and save_checkpoint:
                save_path = Path(
                    os.path.join(dir_checkpoint, args.main_folder, training_model, fitting_model, f'run_{run_number}'))
                Path(save_path).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(save_path / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                wandb.finish(exit_code=0)
            break


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
    parser.add_argument('--patientData', '-dir', type=str, default='/m2_data/mustafa/patientDataReduced/', help='Enther the directory saving the patient data')
    parser.add_argument('--diffusion-direction', '-d', type=str, default='M', help='Enter the diffusion direction: M, I, P or S',
                        dest='dir')
    parser.add_argument('--parallel_training', '-parallel', action='store_true', help='Use argument for parallel training with multiple GPUs.')
    parser.add_argument('--sweep', '-sweep', action='store_true', help='Use this flag if you want to run hyper parameter tuning')
    parser.add_argument('--custom_patient_list', '-clist', type=str, help='Input path to txt file with patient names to be used.')#default='new_patientList.txt'
    parser.add_argument('--input_sigma', '-s',  type=str, help='Use argument if sigma map is used as input.')
    parser.add_argument('--training_model', '-trn', default='attention_unet',help='Specify which training model to use. Choose between ...,...,...')
    parser.add_argument('--fitting_model', '-fit', default='biexp', help='Specify which fitting model to use')
    parser.add_argument('--run_number', '-rnum', default='1', help='This argument is used by sweep_train.py')
    parser.add_argument('--main_folder', '-folder', default='cross_validation_l1', help='Specify main folder name')
    parser.add_argument('--adc_as_loss', '-adc', type=str, help='Pass True if use ADC as loss function')#default='new_patientList.txt'
    parser.add_argument('--use_3D', '-3d', type= str, help='Pass True if using 3D diffusion')#default='new_patientList.txt'
    parser.add_argument('--learn_sigma_scaling', '-ss', type= str, help='Pass True if allowing for AI to learn scaling sigma')#default='new_patientList.txt'
    parser.add_argument('--estimate_S0', '-s0', type= str, help='Pass True if allowing for AI to estimate S0-image')#default='new_patientList.txt'
    parser.add_argument('--feed_sigma', '-fs', type= str, help='Pass True if feeding sigma map to AI. Input sigma has to be true')#default='new_patientList.txt'


    return parser.parse_args()


def main(rank,world_size ,sweep):

    if not sweep:
        #Setup parallel training on multiple GPUs
        setup(rank,world_size)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
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

    b = torch.linspace(0, 2000, steps=21).cuda(non_blocking=True)
    b = b[1:]
    n_channels = 20
    if args.use_3D: n_channels *= 3


    if args.training_model == 'attention_unet':
        n_mess = "atten_unet"
        net = Atten_Unet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, feed_sigma=args.feed_sigma).cuda()
    elif args.training_model == 'unet':
        n_mess = "unet"
        net = UNet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, feed_sigma=args.feed_sigma).cuda()
    elif args.training_model == 'res_atten_unet':
        n_mess = "res_atten_unet"
        net = Res_Atten_Unet(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0, feed_sigma=args.feed_sigma).cuda()
    elif args.training_model == 'unet_2decoder':
        n_mess = "unet_2decoder"
        net = UNet_2Decoders(n_channels=n_channels, rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model, use_3D=args.use_3D, learn_sigma_scaling=args.learn_sigma_scaling, estimate_S0 = args.estimate_S0).cuda()
    elif args.training_model == 'unetr':
        n_mess = "unetr"

        img_size_x = 240
        img_size_y = 208
        patch_size = 16
        in_channels = 20
        embed_dim = in_channels * (patch_size) ** 2
        num_heads = 8
        mlp_ratio = 2
        base_filter = 32 #64 #32 #16
        num_layers = 8


        net = UNETR(img_size_x, img_size_y, patch_size, in_channels, base_filter, embed_dim, num_heads, num_layers,
                      mlp_ratio,rice=True, input_sigma=args.input_sigma, fitting_model=args.fitting_model).cuda()
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
        experiment = wandb.init(mode = 'online')
        config = wandb.config# Automatically pulls sweep parameters from configurations of the sweep
        wandb.run.name = str(f'Batch_size {config.batch_size} num_epochs {config.epochs} lr {config.learning_rate:.4f}')

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
            experiment = wandb.init(project='UNet-Denoise', resume='allow', anonymous='must', mode = 'online')
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
                      training_model = args.training_model,
                      fitting_model = args.fitting_model,
                      run_number= args.run_number
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
                      save_checkpoint=True,
                      training_model = args.training_model,
                      fitting_model = args.fitting_model,
                      run_number = args.run_number
            )
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
