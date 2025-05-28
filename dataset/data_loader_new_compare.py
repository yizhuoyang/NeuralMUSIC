import random
import re
import sys

sys.path.append('../')
from DeepMucis_plus.utlis.util import filter_folders, downsample_audio, normalize_magnitude, normalize_phase, \
    ModeVector_torch,load_dataframe, load_numpy
from torch.utils.data import Dataset
import torchvision.transforms as Trans
import torch
import librosa
from DeepMucis_plus.dataset.data_augmentation import add_gaussian_noise
import pysensing.acoustic.preprocessing.transform as transform
import torch.utils.data as data
import os
import numpy as np
random.seed(0)



class BaseAudioDataset(Dataset):
    def __init__(self, feature='spectrogram', use_real=True, downsample=True, noise_aug=False):
        self.audio_data = []
        self.feature = feature
        self.use_real = use_real
        self.downsample = downsample
        self.noise_aug = noise_aug
        self.stft_trans = transform.stft(n_fft=512, hop_length=256)

    def noise_aug_algo(self, audio_data, snr=None):
        if snr is None:
            if random.random() >= 0.4:
                snr = random.randint(-10, 30)
        if snr is not None:
            audio_data = add_gaussian_noise(audio_data, snr)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        real, imag = stft.real, stft.imag
        magnitude, phase = torch.abs(stft), torch.angle(stft)
        spectrogram = torch.cat([real, imag], dim=0) if self.use_real else torch.cat([magnitude, phase], dim=0)
        target_size = (64, 64) if self.downsample else (257, 64)
        return Trans.Resize(target_size, antialias=True)(spectrogram)


class DAMUSIC_Loader(BaseAudioDataset):
    def __init__(self, root, subset="train", coherent=2, num_source=1, noise_aug=False, model='DAMUSIC', feature='spectrogram', num_percent=1, use_real=True, downsample=True, snr=None, classification=False, mode='all'):
        super().__init__(feature, use_real, downsample, noise_aug)
        self.root = root
        self.model = model
        self.coherent = coherent
        self.num_source = num_source
        self.num_percent = num_percent
        self.snr = snr
        self.classification = classification
        self.mode = mode
        self.transform = 'train' in subset
        subset_data_dirs = []
        for typ, flag in [('coherent', 1), ('incoherent', 0)]:
            if coherent in [flag, 2]:
                path = os.path.join(root, subset, typ)
                if os.path.exists(path):
                    subset_data_dirs += [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]
        if num_source in [1, 2]:
            subset_data_dirs = filter_folders(subset_data_dirs, num_source)
        else:
            subset_data_dirs = [d for d in subset_data_dirs if "NS" in d]

        for seq_path in subset_data_dirs:
            audio_files = os.listdir(seq_path)
            if self.transform:
                audio_files = audio_files[:int(len(audio_files) * num_percent)]
            for file in audio_files:
                if self._filter_mode(file):
                    self.audio_data.append(os.path.join(seq_path, file))

    def _filter_mode(self, filename):
        if self.mode == 'all':
            return True
        match = re.search(r'degree_([-+]?\d*\.?\d+)', filename)
        if not match:
            return False
        deg = float(match.group(1))
        if self.mode == 'range': return 0 <= deg <= 315
        if self.mode == 'step': return 0 <= deg <= 355 and deg % 5 == 0
        if self.mode == 'step_10': return 0 <= deg <= 350 and deg % 10 == 0
        if self.mode == 'gap': return any(base <= deg < base + 20 for base in range(0, 331, 40))
        return False

    def extract_degree_numbers(self, file_path):
        match = re.search(r"degree_([\d.-]+)_+times", os.path.basename(file_path))
        return [float(n) if '.' in n else int(n) for n in match.group(1).split('-')]

    def __getitem__(self, idx):
        path = self.audio_data[idx]
        doas = torch.tensor(self.extract_degree_numbers(path))
        audio_data = torch.from_numpy(np.load(path)).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data, self.snr)
        spectrogram = self.spectrogram_process(audio_data)
        return (spectrogram, doas, torch.tensor(len(doas))) if self.classification else (spectrogram, doas)


class SoClas_database(BaseAudioDataset):
    def __init__(self, root, subset='train', noise_aug=False, feature='spectrogram', num_percent=1.0, downsample=True, use_real=True):
        super().__init__(feature, use_real, downsample, noise_aug)
        self.transform = subset == 'train'
        for class_dir in [os.path.join(root, d) for d in os.listdir(root)]:
            for seq in os.listdir(class_dir):
                full_path = os.path.join(class_dir, seq)
                files = os.listdir(full_path)
                split = int(0.8 * len(files) * num_percent) if self.transform else int(0.8 * len(files))
                selected = files[:split] if self.transform else files[split:]
                self.audio_data += [os.path.join(full_path, f) for f in selected]

    def extract_degree_numbers(self, file_path):
        return [int(re.search(r'class\d+_(\d+)', file_path).group(1))]

    def __getitem__(self, idx):
        path = self.audio_data[idx]
        doas = torch.tensor(self.extract_degree_numbers(path))
        audio_data, _ = librosa.load(path, sr=16000, mono=False)
        audio_data = torch.from_numpy(audio_data).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        return (self.spectrogram_process(audio_data), doas) if self.feature != 'raw' else (downsample_audio(audio_data, 1600), doas)

class AFPILD_raw_Dataset(BaseAudioDataset):
    def __init__(self, dataset_dir, data_type='train', covariant_type='cloth', data_transform='stft', noise_aug=False, num_percent=1, downsample=True, use_real=True):
        super().__init__(data_transform, use_real, downsample, noise_aug)
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)[:int(len(load_dataframe(meta_file)) * num_percent)]
        self.data_dir = dataset_dir

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        audio_data = torch.tensor(load_numpy(os.path.join(self.data_dir, item['raw'])))
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        return self.spectrogram_process(audio_data), torch.tensor([item['loc_azimuth']])


if __name__ == "__main__":
    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
            [ -45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2,  -45.7/1000/2, 0.0],
            [45.7/1000/2,  -45.7/1000/2, 0.0]])
    data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
    dataset   = DAMUSIC_Loader(root=data_dir,subset="train",model='CNN',noise_aug=False,num_percent=0.2,mode='step')
    print(dataset.audio_data)
