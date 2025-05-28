import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.data_loader_new_compare import DAMUSIC_Loader
from model.model_training import ModelTrainer_comparison_cls
from model.Comparison import DeepCNN,DeepMusic,DOAnetSPS_class
import torch.optim as optim
import torch
from utlis.util import RMSPELoss
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def custom_collate_fn(batch):
    spectrograms, doas_list,source_list = zip(*batch)
    spectrograms = torch.stack(spectrograms, dim=0)
    source_list  = torch.stack(source_list,dim=0)
    return spectrograms, list(doas_list),source_list

batch_size = 32
N, T, M = 4,1600,'all'
dataset_zoo = ['simulation','soclas','afpild','rsl']
mic_center= np.c_[[3,3,1]]
epoch = 50
num_worker   = 4
downsmaple = 0
use_real = 1
run_dataset = 0
for num_percent in [0.2,0.4,0.6,0.8,1.0]:

    data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
    train_dataset = DAMUSIC_Loader(root=data_dir,model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,coherent=1,num_percent=num_percent,downsample=downsmaple,use_real=use_real,num_source=M,classification=True)
    val_dataset   = DAMUSIC_Loader(root=data_dir,subset="val",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,coherent=1,downsample=downsmaple,use_real=use_real,num_source=M,classification=True)
    lr = 0.0001
    dropout_rate = 0.6
    use_sigmoid  = False #2 is false
    # model = DeepMusic(input_channel=8,dropout_rate=dropout_rate,use_sigmoid=use_sigmoid).to(device)
    # model = DeepCNN(M,input_channel=8,dropout_rate=dropout_rate).to(device)
    model = DOAnetSPS_class(dropout_rate=dropout_rate).to(device)
    print(val_dataset.__getitem__(0)[0].shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker,collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker,collate_fn=custom_collate_fn)

    criterion = RMSPELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/5-5/saved_{dataset_zoo[run_dataset]}_attention/doanet_{num_percent}_multiple'
    os.makedirs(save_model_dir,exist_ok=True)
    trainer = ModelTrainer_comparison_cls(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
    trainer.train()
