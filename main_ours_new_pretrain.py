import os
from torch import nn
from torch.utils.data import DataLoader
from model.DeepMusic_auto import DeepMusic_plus,DeepMusic_pretrain
from dataset.data_loader_new import DAMUSIC_Loader,DAMUSIC_Loader_pretrain,SoClas_database_pretrain,AFPILD_raw_Dataset_pretrain,RSL_database_pretrain
from model.model_training import ModelTrainer, ModelTrainer_pretrain
import torch.optim as optim
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
run_dataset = 3
dataset_zoo = ['simulation','soclas','afpild','rsl']

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    return inputs, list(targets)

batch_size =256
model = DeepMusic_pretrain().to(device)
for run_dataset in [3]:
    if run_dataset==0:
        data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
        train_dataset = DAMUSIC_Loader_pretrain(root=data_dir,model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,coherent=1,num_source=2)
        val_dataset   = DAMUSIC_Loader_pretrain(root=data_dir,subset="val",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,coherent=1,num_source=2)
    elif run_dataset==1:
        data_dir = '/media/kemove/T9/sound_source_loc/SoClas_database/SoClas_database/Segmented_Sound'
        train_dataset = SoClas_database_pretrain(root=data_dir,noise_aug=True)
        val_dataset   = SoClas_database_pretrain(root=data_dir,subset="val")
    elif run_dataset==2:
        data_dir = '/media/kemove/T9/sound_source_loc/afpild_data'
        train_dataset = AFPILD_raw_Dataset_pretrain(dataset_dir=data_dir,noise_aug=True)
        val_dataset   = AFPILD_raw_Dataset_pretrain(dataset_dir=data_dir,data_type='test')
    elif run_dataset==3:
        data_dir = '/media/kemove/T9/sound_source_loc/locate_dataset/RSL2019'
        train_dataset = RSL_database_pretrain(root=data_dir,noise_aug=True)
        val_dataset   = RSL_database_pretrain(root=data_dir,subset='val')

    print(val_dataset.__getitem__(0)[0].shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=8)

    epoch = 400
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/new_exp/saved_simulation/pretrain_new/{dataset_zoo[run_dataset]}'
    os.makedirs(save_model_dir,exist_ok=True)
    trainer = ModelTrainer_pretrain(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
    trainer.train()
