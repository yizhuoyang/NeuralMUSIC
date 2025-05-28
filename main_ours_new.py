import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model.DeepMusic_auto import DeepMusic_plus
from dataset.data_loader_new import DAMUSIC_Loader,SoClas_database,AFPILD_raw_Dataset,RSL_database
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
dataset_zoo = ['simulation','soclas','afpild','rsl']
mic_center= np.c_[[3,3,1]]
epoch = 100
num_worker   = 8
num_source = M
for mode in  ['all']:
    finetune = False
    for run_dataset in range(1):
        for num_percent in [1.0]:
            for use_pretrain in range(1,2):
                if run_dataset==0:
                    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                    [-45.7/1000/2,  -45.7/1000/2, 0.0],
                    [45.7/1000/2,  -45.7/1000/2, 0.0]])
                    data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
                    train_dataset = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,model='CNN',geometry_aug=False,noise_aug=True,time_aug=False,num_source=num_source,coherent=1,num_percent=num_percent,mode=mode)
                    val_dataset   = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,subset="val",model='CNN',geometry_aug=False,noise_aug=False,time_aug=False,num_source=num_source,coherent=1)
                    lr = 0.001
                    print(train_dataset.audio_data)
                    print(mode)
                elif run_dataset==1:
                    mic_offset = np.array([[ -45.7/1000/2,  45.7/1000/2, 0.0],
                                    [ 45.7/1000/2, 45.7/1000/2, 0.0],
                                    [45.7/1000/2,  -45.7/1000/2, 0.0],
                                    [-45.7/1000/2,  -45.7/1000/2, 0.0]])
                    data_dir = '/media/kemove/T9/sound_source_loc/SoClas_database/SoClas_database/Segmented_Sound'
                    train_dataset = SoClas_database(root=data_dir,mic_offsets=mic_offset,noise_aug=False,num_percent=num_percent)
                    val_dataset   = SoClas_database(root=data_dir,mic_offsets=mic_offset,subset="val")
                    lr = 0.001
                if run_dataset==2:
                    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                    [-45.7/1000/2,  -45.7/1000/2, 0.0],
                    [45.7/1000/2,  -45.7/1000/2, 0.0]])
                    data_dir = '/media/kemove/T9/sound_source_loc/afpild_data'
                    train_dataset = AFPILD_raw_Dataset(dataset_dir=data_dir,mic_offsets=mic_offset,num_percent=num_percent,noise_aug=False)
                    val_dataset   = AFPILD_raw_Dataset(dataset_dir=data_dir,mic_offsets=mic_offset,data_type='test')
                    lr = 0.001
                if run_dataset==3:
                    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
                    [ -45.7/1000/2,  45.7/1000/2, 0.0],
                    [-45.7/1000/2,  -45.7/1000/2, 0.0],
                    [45.7/1000/2,  -45.7/1000/2, 0.0]])
                    data_dir = '/media/kemove/T9/sound_source_loc/locate_dataset/RSL2019'
                    train_dataset = RSL_database(root=data_dir,mic_offsets=mic_offset,num_percent=num_percent,noise_aug=False,subset="train")
                    val_dataset   = RSL_database(root=data_dir,mic_offsets=mic_offset,subset="val")
                    lr = 0.001
                model = DeepMusic_plus(N, T, M,device=device,attention=True).to(device)

                if use_pretrain==1:
                    checkpoint = torch.load(f"/media/kemove/T9/sound_source_loc/simulation_data/new_exp/saved_simulation/pretrain_new/{dataset_zoo[run_dataset]}/last_model")
                    encoder_weights = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if k.startswith("encoder.")}
                    print("loading pretrain weights")
                    model.encoder.load_state_dict(encoder_weights)

                if finetune:
                    # checkpoint = torch.load(f"/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_simulation_attention_new/ours_1.0_1_1/best_model")
                    checkpoint = torch.load(f"/media/kemove/T9/sound_source_loc/simulation_data/3-22/saved_soclas_attention_new/ours_1.0_1/best_model")
                    model.load_state_dict(checkpoint)
                    print("loading finetuning weights")

                print(val_dataset.__getitem__(0)[0].shape,train_dataset.__len__())
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)
                val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_worker)

                criterion = RMSPELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

                scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
                save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_{dataset_zoo[run_dataset]}_attention_new/ours_{num_percent}_{use_pretrain}_{num_source}_noise2'
                # save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/5-6/saved_{dataset_zoo[run_dataset]}_attention/ours_{num_percent}_{use_pretrain}_{mode}'
                # save_model_dir = f'/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_fintune_attention/ours_sclas2sim_{num_percent}'
                os.makedirs(save_model_dir,exist_ok=True)
                trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer,epoch=epoch,model_path=save_model_dir, device=device, lr_scheduler=scheduler)
                trainer.train()
