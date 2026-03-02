import random
import re
import sys
sys.path.append('../')
from dataset.data_loader import load_dataframe, load_numpy
from utlis.util import filter_folders, downsample_audio, normalize_magnitude, normalize_phase, \
    ModeVector_torch, doa_xy_deg_from_xyz, load_gt_dict, stem, CalibFromMat, parse_gt_xyz, doa_xz_deg_from_xyz_cav3d
from torch.utils.data import Dataset
import torchvision.transforms as Trans
import torch
import matplotlib.pyplot as plt
import librosa
from dataset.data_augmentation import add_gaussian_noise
import pysensing.acoustic.preprocessing.transform as transform
import torch.utils.data as data
import os
import re
from pathlib import Path
import numpy as np
import random


class Grid:
    def __init__(self):
        self.x = np.load('/utlis/grid_x.npy')
        self.y = np.load('/utlis/grid_y.npy')
        self.z = np.load('/utlis/grid_z.npy')

def array_aug(mic_offsets,locs,transform,interval=None,mic_center=np.array([[3, 3, 1]])):
    if transform:
        if interval == None:
            random_integer = random.randint(0, 360)
        else:
            random_integer = random.randint(0,int(360/interval))*interval
    else:
        random_integer = 0
    grid = Grid()
    rotation_degree = random_integer
    theta = np.deg2rad(rotation_degree)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])
    rotated_offsets = mic_offsets @ R.T
    mic_locs_rotated = mic_center + rotated_offsets
    mic_positions = mic_locs_rotated.T
    steer_vector_calc = ModeVector_torch(torch.tensor(mic_positions), 16000, 512, 343, grid, "far", precompute=True)
    sv = steer_vector_calc.mode_vec
    return sv,(locs+rotation_degree)%360

class BaseAudioDataset(Dataset):
    def __init__(self, mic_offsets, feature='spectrogram', noise_aug=False, transform_flag=True):
        self.audio_data = []
        self.noise_aug = noise_aug
        self.feature = feature
        self.transform = transform_flag
        self.mic_offsets = mic_offsets
        self.stft_trans = transform.stft(n_fft=512, hop_length=256)
        self.resize_transform = Trans.Resize((257, 64), antialias=True)

    def noise_aug_algo(self, audio_data, snr=None):
        if snr is None:
            if random.random() >= 0.4:
                snr = random.randint(-10, 30)
        if snr is not None:
            audio_data = add_gaussian_noise(audio_data, snr)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = normalize_magnitude(torch.abs(stft))
        phase = normalize_phase(torch.angle(stft))
        spectrogram = torch.stack([magnitude, phase], dim=1).view(2 * stft.shape[0], *stft.shape[1:])
        return self.resize_transform(spectrogram)

    def extract_degree_numbers(self, file_path):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class DAMUSIC_Loader(BaseAudioDataset):
    def __init__(self, root, mic_offsets, subset='train', coherent=2, num_source=1,
                 noise_aug=False, feature='spectrogram', num_percent=1, snr=None, mode='all'):
        super().__init__(mic_offsets, feature, noise_aug, subset == 'train' and mode == 'all')
        self.root = root
        self.num_source = num_source
        self.num_percent = num_percent
        self.snr = snr
        self.mode = mode

        subset_data_dirs = []
        for coh_type in [('coherent', 1), ('incoherent', 0)]:
            if coherent in [coh_type[1], 2]:
                path = os.path.join(root, subset, coh_type[0])
                if os.path.exists(path):
                    subset_data_dirs += [os.path.join(path, d) for d in os.listdir(path) if d.startswith('N')]

        subset_data_dirs = filter_folders(subset_data_dirs, num_source)
        for seq_path in subset_data_dirs:
            audio_files = os.listdir(seq_path)
            if 'train' in subset:
                audio_files = audio_files[:int(len(audio_files) * num_percent)]
            for file in audio_files:
                degree_val = self._extract_degree_from_filename(file)
                if self._filter_by_mode(degree_val):
                    self.audio_data.append(os.path.join(seq_path, file))

    def _extract_degree_from_filename(self, filename):
        match = re.search(r'degree_([-+]?[0-9]*\.?[0-9]+)', filename)
        return float(match.group(1)) if match else None

    def _filter_by_mode(self, degree_val):
        if degree_val is None:
            return False
        if self.mode == 'all':
            return True
        if self.mode == 'range':
            return 0 <= degree_val <= 315 and degree_val % 5 == 0
        if self.mode == 'step':
            return 0 <= degree_val <= 355 and degree_val % 5 == 0
        if self.mode == 'step_10':
            return 0 <= degree_val <= 350 and degree_val % 10 == 0
        if self.mode == 'gap':
            return any(base <= degree_val < base + 20 for base in range(0, 331, 40))
        return False

    def extract_degree_numbers(self, file_path):
        return [float(num) if '.' in num else int(num) for num in re.search(r"degree_([\d.-]+)_+times", os.path.basename(file_path)).group(1).split('-')]

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        doas = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data = torch.from_numpy(np.load(audio_path)).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data, self.snr)
        sv, doas = array_aug(self.mic_offsets, doas, self.transform)
        if self.feature != 'raw':
            return self.spectrogram_process(audio_data), doas, sv
        return audio_data, doas, sv

class SoClas_database(BaseAudioDataset):
    def __init__(self, root, mic_offsets, subset='train', noise_aug=False, feature='spectrogram', num_percent=1):
        super().__init__(mic_offsets, feature, noise_aug, subset == 'train')
        for class_dir in [os.path.join(root, c) for c in os.listdir(root)]:
            file_list = os.listdir(class_dir)
            limit = int(0.8 * len(file_list) * num_percent) if subset == 'train' else int(0.8 * len(file_list))
            files = file_list[:limit] if subset == 'train' else file_list[limit:]
            self.audio_data.extend([os.path.join(class_dir, f) for f in files])

    def extract_degree_numbers(self, file_path):
        return [int(re.search(r'class\d+_(\d+)', file_path).group(1))]

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        doas = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data, _ = librosa.load(audio_path, sr=16000, mono=False)
        audio_data = torch.from_numpy(audio_data).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        sv, doas = array_aug(self.mic_offsets, doas, self.transform)
        if self.feature != 'raw':
            return self.spectrogram_process(audio_data), doas, sv
        return downsample_audio(audio_data, 1600) if self.feature == 'raw' else audio_data, doas, sv

class AFPILD_raw_Dataset(BaseAudioDataset):
    def __init__(self, dataset_dir, mic_offsets, data_type='train', covariant_type='cloth', data_transform='stft', noise_aug=False, num_percent=1):
        super().__init__(mic_offsets, data_transform, noise_aug, data_type == 'train')
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)[:int(len(self.data_arr) * num_percent)]
        self.data_dir = dataset_dir

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        loc_theta = item['loc_azimuth']
        audio = load_numpy(os.path.join(self.data_dir, item['raw']))
        if self.noise_aug:
            audio = self.noise_aug_algo(audio)
        doas = torch.tensor([loc_theta])
        sv, doas = array_aug(self.mic_offsets, doas, self.transform)
        if self.feature != 'raw':
            return self.spectrogram_process(torch.tensor(audio)), doas, sv
        return torch.tensor(audio), doas, sv

class BasePretrainDataset(Dataset):
    def __init__(self, feature='spectrogram', noise_aug=False):
        self.audio_data = []
        self.noise_aug = noise_aug
        self.feature = feature
        self.stft_trans = transform.stft(n_fft=512, hop_length=256)
        self.resize_transform = Trans.Resize((257, 64), antialias=True)

    def noise_aug_algo(self, audio_data):
        if random.random() >= 0.4:
            snr = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, snr)
        return audio_data

    def apply_mic_masking(self, spectrogram):
        C, F, T = spectrogram.shape
        mic_idx = 0
        mag_channel = mic_idx * 2
        phase_channel = mag_channel + 1
        mask = torch.zeros_like(spectrogram[:2], dtype=torch.float32)
        mode = random.choice(["time", "frequency", "patch"])

        if mode == "time":
            span = int(0.1 * T)
            start = random.randint(0, T - span)
            spectrogram[:, :, start:start+span] = 0
            mask[:, :, start:start+span] = 1
        elif mode == "frequency":
            span = int(0.1 * F)
            start = random.randint(0, F - span)
            spectrogram[:, start:start+span, :] = 0
            mask[:, start:start+span, :] = 1
        elif mode == "patch":
            patch_size = 10
            for _ in range(int(0.1 * F * T // (patch_size ** 2))):
                fs = random.randint(0, F - patch_size)
                ts = random.randint(0, T - patch_size)
                spectrogram[:, fs:fs+patch_size, ts:ts+patch_size] = 0
                mask[:, fs:fs+patch_size, ts:ts+patch_size] = 1

        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = normalize_magnitude(torch.abs(stft))
        phase = normalize_phase(torch.angle(stft))
        spectrogram = torch.stack([magnitude, phase], dim=1).view(2 * stft.shape[0], *stft.shape[1:])
        spec_orig = self.resize_transform(spectrogram)
        masked, mask = self.apply_mic_masking(spec_orig.clone())
        return masked, spec_orig[:2], mask

class DAMUSIC_Loader_pretrain(BasePretrainDataset):
    def __init__(self, root, subset="train", coherent=2, num_source=1, 
                 feature='spectrogram', noise_aug=False):
        super().__init__(feature, noise_aug)
        self.root = root

        subset_data_dirs = []
        for coh_type in [('coherent', 1), ('incoherent', 0)]:
            if coherent in [coh_type[1], 2]:
                path = os.path.join(root, subset, coh_type[0])
                if os.path.exists(path):
                    subset_data_dirs += [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]
        subset_data_dirs = filter_folders(subset_data_dirs, num_source)
        for seq_path in subset_data_dirs:
            self.audio_data += [os.path.join(seq_path, f) for f in os.listdir(seq_path)]

    def __getitem__(self, index):
        audio_data = torch.from_numpy(np.load(self.audio_data[index])).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        if self.feature != 'raw':
            return self.spectrogram_process(audio_data)
        return audio_data

class SoClas_database_pretrain(BasePretrainDataset):
    def __init__(self, root, subset='train', feature='spectrogram', noise_aug=False):
        super().__init__(feature, noise_aug)
        for class_dir in [os.path.join(root, d) for d in os.listdir(root)]:
            file_list = os.listdir(class_dir)
            limit = int(0.8 * len(file_list))
            files = file_list[:limit] if subset == 'train' else file_list[limit:]
            for seq in files:
                self.audio_data += [os.path.join(class_dir, seq, f) for f in os.listdir(os.path.join(class_dir, seq))]

    def __getitem__(self, index):
        audio_data, _ = librosa.load(self.audio_data[index], sr=16000, mono=False)
        audio_data = torch.from_numpy(audio_data).float()
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        if self.feature != 'raw':
            return self.spectrogram_process(audio_data)
        return downsample_audio(audio_data, 1600) if self.feature == 'raw' else audio_data

class AFPILD_raw_Dataset_pretrain(BasePretrainDataset):
    def __init__(self, dataset_dir, data_type='train', covariant_type='cloth', data_transform='stft', noise_aug=False):
        super().__init__(data_transform, noise_aug)
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)
        self.data_dir = dataset_dir

    def __getitem__(self, idx):
        audio_data = load_numpy(os.path.join(self.data_dir, self.data_arr[idx]['raw']))
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        return self.spectrogram_process(torch.tensor(audio_data))

class AV16_Dataset(BaseAudioDataset):
    def __init__(
        self,
        processed_root,
        mic_offsets,
        subset='train',
        noise_aug=False,
        feature='spectrogram',
        num_percent=1.0,
        cam=1,
    ):
        super().__init__(
            mic_offsets=mic_offsets,
            feature=feature,
            noise_aug=noise_aug,
            transform_flag=(subset == 'train')
        )

        self.root = Path(processed_root)
        self.subset = subset
        self.num_percent = num_percent
        self.cam = cam

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.audio_data = []  # (audio_path, gt_path)

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue

            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            audio_files = audio_files[:int(len(audio_files) * self.num_percent)]

            for ap in audio_files:
                gp = gt_dir / f"{ap.stem}.npy"
                if gp.exists():
                    self.audio_data.append((ap, gp))

        if not self.audio_data:
            raise RuntimeError(f"No AV16 samples found in {self.root}")

        print(f"[AV16] Indexed {len(self.audio_data)} samples.")

    def extract_degree_numbers(self, file_path):
        raise NotImplementedError("AV16 uses gt3d_xyz instead of filename DOA")

    def __getitem__(self, index):
        ap, gp = self.audio_data[index]

        # -------- Load audio --------
        audio = torch.from_numpy(np.load(str(ap)).astype(np.float32))

        if self.noise_aug:
            audio = self.noise_aug_algo(audio)

        # -------- Load GT --------
        gt = load_gt_dict(gp)
        xyz = np.asarray(gt["gt3d_xyz"], dtype=np.float32)
        loc_theta_deg = np.array(doa_xy_deg_from_xyz(xyz))
        doas = torch.tensor(loc_theta_deg, dtype=torch.float32)

        # -------- Array Aug --------
        sv, doas = array_aug(
            self.mic_offsets,
            doas,
            self.transform,
            mic_center=np.array([0.0, 0.0, 0.0])
        )

        if self.feature != "raw":
            spectrogram = self.spectrogram_process(audio)
            return spectrogram, doas, sv

        return audio, doas, sv

class AV16_Dataset_pretrain(BasePretrainDataset):
    def __init__(
        self,
        processed_root,
        subset='train',
        noise_aug=False,
        feature='spectrogram',
        num_percent=1.0,
        cam=1,
    ):
        super().__init__(feature=feature, noise_aug=noise_aug)

        self.root = Path(processed_root)
        self.subset = subset
        self.num_percent = num_percent
        self.cam = cam

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.audio_data = []

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue

            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            audio_files = audio_files[:int(len(audio_files) * self.num_percent)]

            for ap in audio_files:
                gp = gt_dir / f"{ap.stem}.npy"
                if gp.exists():
                    self.audio_data.append(ap)

        if not self.audio_data:
            raise RuntimeError(f"No AV16 samples found in {self.root}")

        print(f"[AV16-Pretrain] Indexed {len(self.audio_data)} samples.")

    def __getitem__(self, index):
        audio = torch.from_numpy(
            np.load(str(self.audio_data[index])).astype(np.float32)
        )

        if self.noise_aug:
            audio = self.noise_aug_algo(audio)

        if self.feature != "raw":
            return self.spectrogram_process(audio)

        return audio



if __name__ == "__main__":
    index = 300
    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
            [ -45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2,  -45.7/1000/2, 0.0],
            [45.7/1000/2,  -45.7/1000/2, 0.0]])
    root = '/media/kemove/T9/sound_source_loc/simulation_data'
    datset = DAMUSIC_Loader(root,mic_offsets=mic_offset,subset='train',num_percent=0.2,mode='step_10')
    print(datset.audio_data)

