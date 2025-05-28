import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model.DeepMusic_auto import DeepMusic_plus_class
from dataset.data_loader_new_multiple import DAMUSIC_Loader
from model.model_training import ModelTrainer_multiple
import torch.optim as optim
import torch
from utlis.util import RMSPELoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
random.seed(42)

def custom_collate_fn(batch):
    spectrograms, doas_list, sv_list, corr_list,source_list = zip(*batch)
    spectrograms = torch.stack(spectrograms, dim=0)
    correlations = torch.stack(corr_list, dim=0)
    sv_list      = torch.stack(sv_list, dim=0)
    source_list  = torch.stack(source_list,dim=0)
    return spectrograms, list(doas_list), sv_list, correlations,source_list

batch_size = 32
N, T, M = 4,1600,1
dataset_zoo = ['simulation','soclas','afpild','rsl']
mic_center= np.c_[[3,3,1]]
epoch = 50
num_worker   = 8
num_source = M

for num_percent in [0.2,0.4,0.6,0.8,1.0]:
    for use_pretrain in range(2):
        mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
        [ -45.7/1000/2,  45.7/1000/2, 0.0],
        [-45.7/1000/2,  -45.7/1000/2, 0.0],
        [45.7/1000/2,  -45.7/1000/2, 0.0]])
        data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
        train_dataset = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,coherent=1,num_percent=num_percent)
        val_dataset   = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,subset="val",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,coherent=1)
        lr = 0.001
        model = DeepMusic_plus_class(N, T, M,device=device).to(device)

        if use_pretrain==1:
            checkpoint = torch.load(f"/media/kemove/T9/sound_source_loc/simulation_data/new_exp/saved_simulation/pretrain_new/{dataset_zoo[0]}/last_model")
            encoder_weights = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if k.startswith("encoder.")}
            print("loading pretrain weights")
            model.encoder.load_state_dict(encoder_weights)

        print(val_dataset.__getitem__(0)[0].shape,train_dataset.__len__())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker,collate_fn=custom_collate_fn)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker,collate_fn=custom_collate_fn)

        criterion = RMSPELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/5-5/saved_{dataset_zoo[0]}_attention/ours_{num_percent}_{use_pretrain}_multiple'
        os.makedirs(save_model_dir,exist_ok=True)
        trainer = ModelTrainer_multiple(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
        trainer.train()
