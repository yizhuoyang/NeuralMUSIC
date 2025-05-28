import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model.DeepMusic_auto import DeepMusic_plus
from dataset.data_loader_new import DAMUSIC_Loader,SoClas_database,AFPILD_raw_Dataset
from model.model_training import ModelTrainer
import torch.optim as optim
import torch
from utlis.util import RMSPELoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    return inputs, list(targets)

batch_size = 32
N, T, M = 4,1600,1
dataset_zoo = ['simulation','soclas','afpild']
mic_center= np.c_[[3,3,1]]
epoch = 50
num_worker   = 8
num_source = M


mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
[ -45.7/1000/2,  45.7/1000/2, 0.0],
[-45.7/1000/2,  -45.7/1000/2, 0.0],
[45.7/1000/2,  -45.7/1000/2, 0.0]])
data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'

for room in ['C']:
    train_dataset = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,subset=f"train_room{room}",model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,num_source=num_source,coherent=1,num_percent=0.1)
    val_dataset   = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,subset=f"val_room{room}",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,num_source=num_source,coherent=1)
    lr = 0.001

    model = DeepMusic_plus(N, T, M,device=device).to(device)

    checkpoint = torch.load(f"/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_simulation_attention_new/ours_1.0_1_1/best_model")
    model.load_state_dict(checkpoint)
    print("loading pretrain weights")
    #
    for param in model.convs1.parameters():
        param.requires_grad = False

    for param in model.convs2.parameters():
        param.requires_grad = False

    for param in model.channel_attention.parameters():
        param.requires_grad = False
    #
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    print(val_dataset.__getitem__(0)[0].shape,val_dataset.__len__(),train_dataset.__len__())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)

    criterion = RMSPELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/5-13/saved_simulation/ours_finetune_{room}'
    os.makedirs(save_model_dir,exist_ok=True)
    trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
    trainer.train()
