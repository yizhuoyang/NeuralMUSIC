import os
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from model.DeepMusic_auto import DeepMusic_plus
from dataset.data_loader_new_compare import DAMUSIC_Loader,SoClas_database,AFPILD_raw_Dataset,RSL_database
from model.model_training import ModelTrainer_comparison,ModelTrainer
from model.Comparison import DeepCNN,DeepMusic,DOAnetSPS
from model.Comparison import DAMUSIC
import torch.optim as optim
import torch
from utlis.util import RMSPELoss
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 32
N, T, M = 4,1600,1
dataset_zoo = ['simulation','soclas','afpild','rsl']
mic_center= np.c_[[3,3,1]]
epoch = 100
num_worker   = 8
downsmaple = 1
num_source = M
use_real = 0
for mode in ['step']:
    for run_dataset in range(1):
        percent_list = [1.0]
        for num_percent in percent_list:
            for use_pretrain in range(1):
                if run_dataset==0:
                    mic_center= np.c_[[3,3,1]]
                    mic_locs = mic_center + np.c_[[ 45.7/1000/2, 45.7/1000/2, 0.0],
                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                    [-45.7/1000/2,  -45.7/1000/2, 0.0],
                    [45.7/1000/2,  -45.7/1000/2, 0.0]]
                    use_real = 1
                    data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
                    train_dataset = DAMUSIC_Loader(root=data_dir,model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,coherent=1,num_percent=num_percent,num_source=num_source,downsample=downsmaple,use_real=use_real,mode=mode)
                    val_dataset   = DAMUSIC_Loader(root=data_dir,subset="val",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,coherent=1,num_source=num_source,downsample=downsmaple,use_real=use_real)
                    lr = 0.0001
                    print(train_dataset.audio_data)
                elif run_dataset==1:
                    mic_center= np.c_[[3,3,1]]
                    mic_locs = mic_center + np.c_[
                                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                                    [ 45.7/1000/2, 45.7/1000/2, 0.0],
                                    [45.7/1000/2,  -45.7/1000/2, 0.0],
                                    [-45.7/1000/2,  -45.7/1000/2, 0.0]]
                    data_dir = '/media/kemove/T9/sound_source_loc/SoClas_database/SoClas_database/Segmented_Sound'
                    train_dataset = SoClas_database(root=data_dir,noise_aug=False,num_percent=num_percent,use_real=False)
                    val_dataset   = SoClas_database(root=data_dir,subset="val",use_real=False)
                    lr = 0.001
                if run_dataset==2:
                    mic_center= np.c_[[3,3,1]]
                    mic_locs = mic_center + np.c_[[ 45.7/1000/2, 45.7/1000/2, 0.0],
                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                    [-45.7/1000/2,  -45.7/1000/2, 0.0],
                    [45.7/1000/2,  -45.7/1000/2, 0.0]]
                    data_dir = '/media/kemove/T9/sound_source_loc/afpild_data'
                    train_dataset = AFPILD_raw_Dataset(dataset_dir=data_dir,num_percent=num_percent,noise_aug=False,use_real=False)
                    val_dataset   = AFPILD_raw_Dataset(dataset_dir=data_dir,data_type='test',use_real=False)
                    lr = 0.001

            if run_dataset==3:
                mic_center= np.c_[[3,3,1]]
                mic_locs = mic_center + np.c_[[ 45.7/1000/2, 45.7/1000/2, 0.0],
                [ -45.7/1000/2,  45.7/1000/2, 0.0],
                [-45.7/1000/2,  -45.7/1000/2, 0.0],
                [45.7/1000/2,  -45.7/1000/2, 0.0]]
                data_dir = '/media/kemove/T9/sound_source_loc/locate_dataset/RSL2019'
                train_dataset = RSL_database(root=data_dir,num_percent=num_percent,noise_aug=False,subset="train",downsample=downsmaple,use_real=use_real)
                val_dataset   = RSL_database(root=data_dir,subset="val",downsample=downsmaple,use_real=use_real)
                lr = 0.0001
                dropout_rate = 0.1
                use_sigmoid  = False
                input_channel = 8

            model = DAMUSIC(N, T, M,mic_positions=mic_locs.T,device=device).to(device)
            print(val_dataset.__getitem__(0)[0].shape)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)
            criterion = RMSPELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            # scheduler = MultiStepLR(optimizer, milestones=[20,60], gamma=0.1)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            # save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_{dataset_zoo[run_dataset]}_attention/DAMUSIC_{num_percent}_{num_source}'
            save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/5-6/saved_{dataset_zoo[run_dataset]}_attention/DAMUSIC_{num_percent}_{mode}'
            os.makedirs(save_model_dir,exist_ok=True)
            trainer = ModelTrainer_comparison(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
            trainer.train()
