import random
import re
import sys
from collections import defaultdict
sys.path.append('../')
from DeepMucis_plus.utlis.util import filter_folders, downsample_audio, normalize_magnitude, normalize_phase, \
    ModeVector_torch,array_aug,load_dataframe, load_numpy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as Trans
import torch
import matplotlib.pyplot as plt
import librosa
from DeepMucis_plus.dataset.data_augmentation import add_gaussian_noise
import pysensing.acoustic.preprocessing.transform as transform
import torch.utils.data as data
import os
import pandas as pd
import numpy as np
import random
random.seed(0)


class DAMUSIC_Loader(Dataset):
    def __init__(self, root,mic_offsets, subset = "train",coherent=2,noise_aug=False,time_aug=False,geometry_aug=False,model='DAMUSIC',feature='spectrogram',num_percent=1,snr=None)-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.geometry_aug = geometry_aug
        self.model = model
        self.coherent = coherent
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        self.snr = snr
        if 'train' not in subset:
            self.transform = False
        else:
            self.transform = True
        subset_data_dirs = []

        if coherent in [1, 2]:
            coherent_path = os.path.join(self.root, subset, 'coherent')
            if os.path.exists(coherent_path):
                subset_data_dirs += [os.path.join(coherent_path, d) for d in os.listdir(coherent_path)
                                     if os.path.isdir(os.path.join(coherent_path, d))]

        if coherent in [0, 2]:
            incoherent_path = os.path.join(self.root, subset, 'incoherent')
            if os.path.exists(incoherent_path):
                subset_data_dirs += [os.path.join(incoherent_path, d) for d in os.listdir(incoherent_path)
                                     if os.path.isdir(os.path.join(incoherent_path, d))]

        subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]
        subset_data_dirs = [folder for folder in subset_data_dirs if f"NS" in folder]

        for seq_path in subset_data_dirs:
            audio_file_list = os.listdir(seq_path)
            if 'train' in subset:
                audio_file_list = audio_file_list[:int(len(audio_file_list)*self.num_percent)]
            for audio_file in audio_file_list:
                self.audio_data.append(os.path.join(seq_path, audio_file))

        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def extract_degree_numbers(self,file_path):
        file_name = os.path.basename(file_path)  # Extract file name
        match = re.search(r"degree_([\d.-]+)_+times", file_name)  # Handle both single and double underscores
        degree_part = match.group(1)  # Extract the matched part
        number_list = [float(num) if '.' in num else int(num) for num in degree_part.split('-')]  # Convert numbers
        return number_list

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram

    def noise_aug_algo(self,audio_data):
        if self.snr==None:
            rand_num_noise = random.random()
            if rand_num_noise>=0.4:
                target_snr_db = random.randint(-10,30)
                audio_data = add_gaussian_noise(audio_data,target_snr_db)
        else:
            target_snr_db = self.snr
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        doas       = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data  = np.load(audio_path)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        sv,doas = array_aug(self.mic_offsets,doas,self.transform)
        num_source = torch.tensor(len(doas))
        spectrogram = self.spectrogram_process(audio_data)
        return spectrogram,doas,sv,num_source




if __name__ == "__main__":
    def custom_collate_fn(batch):
        spectrograms, doas_list, sv_list,source_list = zip(*batch)
        spectrograms = torch.stack(spectrograms, dim=0)
        sv_list      = torch.stack(sv_list, dim=0)
        source_list  = torch.stack(source_list,dim=0)
        return spectrograms, list(doas_list), sv_list,source_list

    index = 300
    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
            [ -45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2,  -45.7/1000/2, 0.0],
            [45.7/1000/2,  -45.7/1000/2, 0.0]])

    root = '/media/kemove/T9/sound_source_loc/simulation_data'
    dataset = DAMUSIC_Loader(root,mic_offset,subset='val',noise_aug=True)
    s,d,sv,num_source = dataset.__getitem__(18861)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    for i, (spectrograms, doas, svs,num_source) in enumerate(train_loader):
        print(f"Batch {i}")
        print(f"  Spectrograms shape: {spectrograms.shape}")   # torch.Size([B, C, F, T])
        print(f"  DOAs: {doas}")                              # list of tensors or lists (variable length)
        print(f"  SVs: {svs.shape}")                           # list of steering vectors
        print(f"  Number Source: {num_source}")   # torch.Size([B, C, C, F])
        if i == 1:  # Just show two batches
            break
