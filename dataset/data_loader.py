import random
import re
import sys
from collections import defaultdict
from pathlib import Path
sys.path.append('../')
# from dataset.data_loader import 
from utlis.util import filter_folders, downsample_audio, normalize_magnitude, normalize_phase, \
    ModeVector_torch, doa_xy_deg_from_xyz, load_gt_dict, stem, CalibFromMat, parse_gt_xyz, doa_xz_deg_from_xyz_cav3d,load_dataframe, load_numpy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as Trans
import torch
import matplotlib.pyplot as plt
import librosa
from dataset.data_augmentation import add_gaussian_noise
import pysensing.acoustic.preprocessing.transform as transform
import torch.utils.data as data
import os
import re
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
from collections import defaultdict
import soundfile as sf
import scipy.io as sio
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def compute_correlation_matrices_torch(X: torch.Tensor) -> torch.Tensor:
    """
    X: (C, F, T) complex
    Return: (F, C, C), mean over time frames (T)
    """
    X = X.permute(2, 1, 0)  # (T, F, C)
    C_hat = torch.matmul(X.unsqueeze(-1), X.unsqueeze(-2).conj())  # (T,F,C,C)
    C_hat = C_hat.mean(dim=0)  # (F,C,C)
    return C_hat


class Grid:
    """
    Cache grid arrays to avoid repeated disk I/O.
    """
    _cached = None

    def __new__(cls, grid_dir="/home/kemove/yyz/SubspaceNet/DeepMucis_plus"):
        if cls._cached is None:
            obj = super().__new__(cls)
            obj.x = np.load(os.path.join(grid_dir, "grid_x.npy"))
            obj.y = np.load(os.path.join(grid_dir, "grid_y.npy"))
            obj.z = np.load(os.path.join(grid_dir, "grid_z.npy"))
            cls._cached = obj
        return cls._cached


# =========================
# Shared augmentation blocks
# =========================

@dataclass
class NoiseAugConfig:
    enabled: bool = False
    p: float = 0.5
    snr_db: Optional[int] = None          # None -> random
    snr_range: Tuple[int, int] = (-10, 30)

def apply_noise_aug(audio_np: np.ndarray, cfg: NoiseAugConfig) -> np.ndarray:
    """
    audio_np: (C,T) or (T,) numpy
    """
    if not cfg.enabled:
        return audio_np
    if random.random() < cfg.p:
        snr = cfg.snr_db if cfg.snr_db is not None else random.randint(*cfg.snr_range)
        audio_np = add_gaussian_noise(audio_np, snr)
    return audio_np


@dataclass
class ArrayAugConfig:
    enabled: bool = True
    interval: Optional[int] = None          # None -> any degree, else e.g. 5/10
    mic_center: np.ndarray = np.array([[3, 3, 1]], dtype=np.float32)
    grid_dir: str = "/home/kemove/yyz/SubspaceNet/DeepMucis_plus"

class ArrayAugmentor:
    """
    Rotation augmentation:
    - rotate mic offsets around Z
    - compute steering vectors (sv)
    - shift doas
    """
    def __init__(self, mic_offsets: np.ndarray, cfg: ArrayAugConfig):
        self.mic_offsets = mic_offsets.astype(np.float32)
        self.cfg = cfg
        self.grid = Grid(cfg.grid_dir)

    def _sample_rotation(self, do_transform: bool) -> int:
        if not (self.cfg.enabled and do_transform):
            return 0
        if self.cfg.interval is None:
            return random.randint(0, 360)
        kmax = int(360 / self.cfg.interval)
        return random.randint(0, kmax) * self.cfg.interval

    def __call__(self, doas_deg: Union[torch.Tensor, np.ndarray, float, int], do_transform: bool):
        rot = self._sample_rotation(do_transform)
        theta = np.deg2rad(rot)

        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1],
            ],
            dtype=np.float32
        )

        rotated_offsets = self.mic_offsets @ R.T
        mic_locs = self.cfg.mic_center + rotated_offsets
        mic_positions = mic_locs.T  # (3, C)

        steer_vector_calc = ModeVector_torch(
            torch.tensor(mic_positions), 16000, 512, 343, self.grid, "far", precompute=True
        )
        sv = steer_vector_calc.mode_vec

        if not torch.is_tensor(doas_deg):
            doas_deg = torch.tensor(doas_deg, dtype=torch.float32)
        doas_deg = (doas_deg + rot) % 360
        return sv, doas_deg


# =========================
# Shared feature extractor
# =========================

@dataclass
class SpecConfig:
    n_fft: int = 512
    hop: int = 256
    resize_hw: Tuple[int, int] = (257, 64)
    return_corr: bool = True
    mode: str = "magphase"  # "magphase" or "cat_mag_phase" (compat)

class SpectrogramExtractor:
    """
    Default: produce (2C, F, T) using [magnitude, phase] per mic stacked then reshaped.
    """
    def __init__(self, cfg: SpecConfig):
        self.cfg = cfg
        self.stft = transform.stft(n_fft=cfg.n_fft, hop_length=cfg.hop)
        self.resize = Trans.Resize(cfg.resize_hw, antialias=True)

    def __call__(self, audio_t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        audio_t: (C,T) float
        return:
          spec: (2C, 257, 64) after resize
          corr: (F, C, C) complex-ish (torch complex) or None
        """
        stft = self.stft(audio_t)  # (C,F,T) complex
        corr = compute_correlation_matrices_torch(stft) if self.cfg.return_corr else None

        mag = torch.abs(stft)
        mag = normalize_magnitude(mag)

        ph = torch.angle(stft)
        ph = normalize_phase(ph)

        if self.cfg.mode == "magphase":
            spec = torch.stack([mag, ph], dim=1)      # (C,2,F,T)
            spec = spec.view(2 * spec.shape[0], spec.shape[2], spec.shape[3])  # (2C,F,T)
        else:
            # legacy: concat along channel dimension
            spec = torch.cat([mag, ph], dim=0)  # (2C,F,T)

        spec = self.resize(spec)
        return spec, corr


@dataclass
class MaskConfig:
    enabled: bool = True
    p: float = 1.0
    ratio: float = 0.1
    patch_size: int = 10

class MicMasker:
    """
    Mask only mic_idx=0 channels (mag+phase) like your original code.
    """
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg

    def __call__(self, spec_2c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        spec_2c: (2C,F,T)
        return:
          masked_spec: (2C,F,T)
          ori_first_mic: (2,F,T)
          mask: (2,F,T)  1 where masked
        """
        ori = spec_2c[:2].clone()
        mask = torch.zeros_like(ori, dtype=torch.float32)

        if (not self.cfg.enabled) or (random.random() > self.cfg.p):
            return spec_2c, ori, mask

        _, F, T = spec_2c.shape
        mag_ch, ph_ch = 0, 1
        mode = random.choice(["time", "frequency", "patch"])

        if mode == "time":
            n = max(1, int(self.cfg.ratio * T))
            s = random.randint(0, max(0, T - n))
            spec_2c[mag_ch, :, s:s+n] = 0
            spec_2c[ph_ch,  :, s:s+n] = 0
            mask[0, :, s:s+n] = 1
            mask[1, :, s:s+n] = 1

        elif mode == "frequency":
            n = max(1, int(self.cfg.ratio * F))
            s = random.randint(0, max(0, F - n))
            spec_2c[mag_ch, s:s+n, :] = 0
            spec_2c[ph_ch,  s:s+n, :] = 0
            mask[0, s:s+n, :] = 1
            mask[1, s:s+n, :] = 1

        else:
            ps = self.cfg.patch_size
            n_patches = max(1, int(self.cfg.ratio * (F * T) / (ps * ps)))
            for _ in range(n_patches):
                fs = random.randint(0, max(0, F - ps))
                ts = random.randint(0, max(0, T - ps))
                spec_2c[mag_ch, fs:fs+ps, ts:ts+ps] = 0
                spec_2c[ph_ch,  fs:fs+ps, ts:ts+ps] = 0
                mask[0, fs:fs+ps, ts:ts+ps] = 1
                mask[1, fs:fs+ps, ts:ts+ps] = 1

        return spec_2c, ori, mask


class BaseAudioDataset(torch.utils.data.Dataset):
    """
    Generic SSL dataset (returns sv + corr by default).
    Subclasses should implement:
      - _build_items()
      - _load_audio_np(item) -> np.ndarray (C,T)
      - _load_doas(item) -> torch.Tensor (K,)
    """
    def __init__(
        self,
        subset: str,
        feature: str = "spectrogram",
        model: str = "DAMUSIC",
        noise_cfg: NoiseAugConfig = NoiseAugConfig(False),
        spec_extractor: Optional[SpectrogramExtractor] = None,
        array_aug: Optional[ArrayAugmentor] = None,
        transform_when_train: bool = True,
    ):
        super().__init__()
        self.subset = subset
        self.feature = feature
        self.model = model

        self.noise_cfg = noise_cfg
        self.spec_extractor = spec_extractor
        self.array_aug = array_aug

        self.do_transform = (transform_when_train and ("train" in subset))
        self.items: List[Any] = []
        self._build_items()

        if len(self.items) == 0:
            raise RuntimeError(f"[{self.__class__.__name__}] Empty items list.")

    def _build_items(self):
        raise NotImplementedError

    def _load_audio_np(self, item: Any) -> np.ndarray:
        raise NotImplementedError

    def _load_doas(self, item: Any) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        audio_np = self._load_audio_np(item)
        audio_np = apply_noise_aug(audio_np, self.noise_cfg)
        audio_t = torch.from_numpy(audio_np).float()

        doas = self._load_doas(item)

        assert self.array_aug is not None, "This dataset requires array_aug (sv computation)."
        sv, doas = self.array_aug(doas, self.do_transform)

        if self.feature != "raw":
            assert self.spec_extractor is not None
            spec, corr = self.spec_extractor(audio_t)
            return spec, doas, sv, corr
        else:
            audio_out = audio_t
            if self.model == "DAMUSIC":
                audio_out = downsample_audio(audio_out, 1600)
            return audio_out, doas, sv


class BasePretrainDataset(torch.utils.data.Dataset):
    """
    Pretrain dataset: returns (masked_spec, ori_first_mic, mask)
    Subclasses implement:
      - _build_items()
      - _load_audio_np(item) -> np.ndarray (C,T)
    """
    def __init__(
        self,
        subset: str,
        feature: str = "spectrogram",
        noise_cfg: NoiseAugConfig = NoiseAugConfig(False),
        spec_extractor: Optional[SpectrogramExtractor] = None,
        masker: Optional[MicMasker] = None,
        model: str = "DAMUSIC",
    ):
        super().__init__()
        self.subset = subset
        self.feature = feature
        self.model = model

        self.noise_cfg = noise_cfg
        self.spec_extractor = spec_extractor
        self.masker = masker

        self.items: List[Any] = []
        self._build_items()
        if len(self.items) == 0:
            raise RuntimeError(f"[{self.__class__.__name__}] Empty items list.")

    def _build_items(self):
        raise NotImplementedError

    def _load_audio_np(self, item: Any) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio_np = self._load_audio_np(item)
        audio_np = apply_noise_aug(audio_np, self.noise_cfg)
        audio_t = torch.from_numpy(audio_np).float()

        if self.feature == "raw":
            # rare case: raw pretrain
            audio_out = audio_t
            if self.model == "DAMUSIC":
                audio_out = downsample_audio(audio_out, 1600)
            return audio_out

        assert self.spec_extractor is not None
        assert self.masker is not None

        spec, _ = self.spec_extractor(audio_t)  # return_corr should be False for speed
        spec_masked, ori, mask = self.masker(spec.clone())
        return spec_masked, ori, mask



class DAMUSIC_Loader(BaseAudioDataset):
    def __init__(
        self,
        root: str,
        mic_offsets: np.ndarray,
        subset: str = "train",
        coherent: int = 2,
        num_source: int = 1,
        noise_aug: bool = False,
        time_aug: bool = False,
        geometry_aug: bool = False,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        num_percent: float = 1.0,
        snr: Optional[int] = None,
        mode: str = "all",
    ):
        self.root = str(root)
        self.coherent = coherent
        self.num_source = num_source
        self.time_aug = time_aug
        self.geometry_aug = geometry_aug
        self.num_percent = float(num_percent)
        self.mode = mode

        # original behavior: if not train or mode!=all => transform False
        transform_when_train = True
        if ("train" not in subset) or (mode != "all"):
            transform_when_train = False

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.6, snr_db=snr, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=True, mode="magphase"))
        array_aug = ArrayAugmentor(
            mic_offsets,
            ArrayAugConfig(enabled=True, interval=None, mic_center=np.array([[3, 3, 1]], dtype=np.float32))
        )

        super().__init__(
            subset=subset,
            feature=feature,
            model=model,
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            array_aug=array_aug,
            transform_when_train=transform_when_train,
        )

    def _build_items(self):
        subset = self.subset
        subset_data_dirs: List[str] = []

        # coherent / incoherent folders
        if self.coherent in [1, 2]:
            coherent_path = os.path.join(self.root, subset, "coherent")
            if os.path.exists(coherent_path):
                subset_data_dirs += [
                    os.path.join(coherent_path, d)
                    for d in os.listdir(coherent_path)
                    if os.path.isdir(os.path.join(coherent_path, d))
                ]

        if self.coherent in [0, 2]:
            incoherent_path = os.path.join(self.root, subset, "incoherent")
            if os.path.exists(incoherent_path):
                subset_data_dirs += [
                    os.path.join(incoherent_path, d)
                    for d in os.listdir(incoherent_path)
                    if os.path.isdir(os.path.join(incoherent_path, d))
                ]

        # geometry aug off => only "N..."
        if not self.geometry_aug:
            subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]

        subset_data_dirs = filter_folders(subset_data_dirs, self.num_source)

        items: List[str] = []
        for seq_path in subset_data_dirs:
            audio_file_list = sorted(os.listdir(seq_path))
            if "train" in subset:
                audio_file_list = audio_file_list[: int(len(audio_file_list) * self.num_percent)]

            for audio_file in audio_file_list:
                full = os.path.join(seq_path, audio_file)

                if self.mode == "all":
                    items.append(full)
                    continue

                # parse degree in filename
                match = re.search(r"degree_([-+]?\d*\.?\d+)", audio_file)
                if not match:
                    continue
                degree_val = float(match.group(1))

                if self.mode == "range":
                    if 0 <= degree_val <= 315 and abs(degree_val - round(degree_val / 5) * 5) < 1e-3:
                        items.append(full)
                elif self.mode == "step":
                    if 0 <= degree_val <= 355 and abs(degree_val - round(degree_val / 5) * 5) < 1e-3:
                        items.append(full)
                elif self.mode == "step_10":
                    if 0 <= degree_val <= 350 and abs(degree_val - round(degree_val / 10) * 10) < 1e-3:
                        items.append(full)
                elif self.mode == "gap":
                    for base in range(0, 331, 40):
                        if base <= degree_val < base + 20:
                            items.append(full)

        self.items = items

    @staticmethod
    def _extract_degree_numbers(file_path: str) -> List[float]:
        file_name = os.path.basename(file_path)
        match = re.search(r"degree_([\d.-]+)_+times", file_name)
        if match is None:
            raise ValueError(f"Cannot parse degree from: {file_path}")
        degree_part = match.group(1)
        nums = [float(num) if "." in num else int(num) for num in degree_part.split("-")]
        return nums

    def _load_audio_np(self, item: str) -> np.ndarray:
        # your sim data is npy
        audio = np.load(item)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio.astype(np.float32)

    def _load_doas(self, item: str) -> torch.Tensor:
        doas = torch.tensor(self._extract_degree_numbers(item), dtype=torch.float32)
        return doas


class GSP_Loader_pretrain(BasePretrainDataset):
    """
    Pretrain on simulation DAMUSIC data: returns masked spec + ori mic0 spec + mask
    """
    def __init__(
        self,
        root: str,
        subset: str = "train",
        coherent: int = 2,
        num_source: int = 1,
        noise_aug: bool = False,
        time_aug: bool = False,
        geometry_aug: bool = False,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        num_mic: int = 4,
    ):
        self.root = str(root)
        self.coherent = coherent
        self.num_source = num_source
        self.time_aug = time_aug
        self.geometry_aug = geometry_aug
        self.num_mic = num_mic

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.6, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=False, mode="magphase"))
        masker = MicMasker(MaskConfig(enabled=True, p=1.0, ratio=0.1, patch_size=10))

        super().__init__(
            subset=subset,
            feature=feature,
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            masker=masker,
            model=model,
        )

    def _build_items(self):
        subset = self.subset
        subset_data_dirs: List[str] = []

        if self.coherent in [1, 2]:
            coherent_path = os.path.join(self.root, subset, "coherent")
            if os.path.exists(coherent_path):
                subset_data_dirs += [
                    os.path.join(coherent_path, d)
                    for d in os.listdir(coherent_path)
                    if os.path.isdir(os.path.join(coherent_path, d))
                ]

        if self.coherent in [0, 2]:
            incoherent_path = os.path.join(self.root, subset, "incoherent")
            if os.path.exists(incoherent_path):
                subset_data_dirs += [
                    os.path.join(incoherent_path, d)
                    for d in os.listdir(incoherent_path)
                    if os.path.isdir(os.path.join(incoherent_path, d))
                ]

        if not self.geometry_aug:
            subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]

        subset_data_dirs = filter_folders(subset_data_dirs, self.num_source)

        items: List[str] = []
        for seq_path in subset_data_dirs:
            for audio_file in sorted(os.listdir(seq_path)):
                items.append(os.path.join(seq_path, audio_file))
        self.items = items

    def _load_audio_np(self, item: str) -> np.ndarray:
        audio = np.load(item)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio.astype(np.float32)

class SoClas_database_pretrain(BasePretrainDataset):
    """
    Pretrain SoClas: returns masked spec + ori mic0 spec + mask
    """
    def __init__(
        self,
        root: str,
        subset: str = "train",
        noise_aug: bool = False,
        time_aug: bool = False,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        sr: int = 16000,
    ):
        self.root = str(root)
        self.time_aug = time_aug
        self.sr = int(sr)

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.6, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=False, mode="magphase"))
        masker = MicMasker(MaskConfig(enabled=True, p=1.0, ratio=0.1, patch_size=10))

        super().__init__(
            subset=subset,
            feature=feature,
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            masker=masker,
            model=model,
        )

    def _build_items(self):
        items: List[str] = []
        all_class_list = [os.path.join(self.root, audio_class) for audio_class in os.listdir(self.root)]
        for class_dir in all_class_list:
            if not os.path.isdir(class_dir):
                continue
            subset_data_dir = os.listdir(class_dir)
            for seq in subset_data_dir:
                subseq = os.path.join(class_dir, seq)
                if not os.path.isdir(subseq):
                    continue
                audio_files = sorted(os.listdir(subseq))
                if self.subset == "train":
                    pick = audio_files[: int(0.8 * len(audio_files))]
                else:
                    pick = audio_files[int(0.8 * len(audio_files)) :]
                for f in pick:
                    items.append(os.path.join(subseq, f))
        self.items = items

    def _load_audio_np(self, item: str) -> np.ndarray:
        audio, _ = librosa.load(item, sr=self.sr, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio.astype(np.float32)

class SoClas_database(BaseAudioDataset):
    """
    SoClas supervised SSL dataset:
    - parse doa from filename: class\d+_(\d+)
    - returns spec/doas/sv/corr or raw/doas/sv
    """
    def __init__(
        self,
        root: str,
        mic_offsets: np.ndarray,
        subset: str = "train",
        noise_aug: bool = False,
        time_aug: bool = False,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        num_percent: float = 1.0,
        sr: int = 16000,
    ):
        self.root = Path(root)
        self.time_aug = time_aug
        self.num_percent = float(num_percent)
        self.sr = int(sr)

        # keep your old behavior: train transform=False
        transform_when_train = False

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.5, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=True, mode="magphase"))
        array_aug = ArrayAugmentor(
            mic_offsets,
            ArrayAugConfig(enabled=True, interval=5, mic_center=np.array([[3, 3, 1]], dtype=np.float32))
        )

        super().__init__(
            subset=subset,
            feature=feature,
            model=model,
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            array_aug=array_aug,
            transform_when_train=transform_when_train,
        )

    def _build_items(self):
        items: List[str] = []
        all_class_list = [self.root / d for d in os.listdir(self.root)]
        for class_dir in all_class_list:
            if not class_dir.is_dir():
                continue
            for seq in sorted(os.listdir(class_dir)):
                subseq = class_dir / seq
                if not subseq.is_dir():
                    continue
                audio_files = sorted(os.listdir(subseq))
                if self.subset == "train":
                    end = int(0.8 * len(audio_files) * self.num_percent)
                    pick = audio_files[:end]
                else:
                    start = int(0.8 * len(audio_files))
                    pick = audio_files[start:]
                for f in pick:
                    items.append(str(subseq / f))
        self.items = items

    @staticmethod
    def _parse_doa(file_path: str) -> List[int]:
        m = re.search(r"class\d+_(\d+)", file_path)
        if m is None:
            raise ValueError(f"Cannot parse doa from path: {file_path}")
        return [int(m.group(1))]

    def _load_audio_np(self, item: str) -> np.ndarray:
        audio, _ = librosa.load(item, sr=self.sr, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio.astype(np.float32)

    def _load_doas(self, item: str) -> torch.Tensor:
        return torch.tensor(self._parse_doa(item), dtype=torch.float32)


class AFPILD_raw_Dataset_pretrain(BasePretrainDataset):
    def __init__(
        self,
        dataset_dir: str,
        data_type: str = "train",
        covariant_type: str = "cloth",
        data_transform: str = "stft",
        noise_aug: bool = False,
    ):
        self.data_dir = str(dataset_dir)
        meta_file = os.path.join(self.data_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)
        self.data_transform = data_transform

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.5, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=False, mode="magphase"))
        masker = MicMasker(MaskConfig(enabled=True, p=1.0, ratio=0.1, patch_size=10))

        super().__init__(
            subset=data_type,
            feature=("raw" if data_transform == "raw" else "spectrogram"),
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            masker=masker,
            model="DAMUSIC",
        )

    def _build_items(self):
        # store index pointers
        self.items = list(range(len(self.data_arr)))

    def _load_audio_np(self, item_idx: int) -> np.ndarray:
        item = self.data_arr[item_idx]
        audio = load_numpy(os.path.join(self.data_dir, item["raw"]))  # expected (C,T)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio

class AFPILD_raw_Dataset(BaseAudioDataset):
    def __init__(
        self,
        dataset_dir: str,
        mic_offsets: np.ndarray,
        data_type: str = "train",
        covariant_type: str = "cloth",
        data_transform: str = "stft",
        noise_aug: bool = False,
        num_percent: float = 1.0,
    ):
        self.data_dir = str(dataset_dir)
        meta_file = os.path.join(self.data_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)
        self.data_arr = self.data_arr[: int(len(self.data_arr) * float(num_percent))]
        self.data_transform = data_transform

        transform_when_train = True

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.5, snr_db=None, snr_range=(-10, 30))

        # Keep your AFPILD custom spectrogram variant: concat [mag, phase] rather than stack->view
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=True, mode="cat_mag_phase"))

        array_aug = ArrayAugmentor(
            mic_offsets,
            ArrayAugConfig(enabled=True, interval=None, mic_center=np.array([[3, 3, 1]], dtype=np.float32))
        )

        super().__init__(
            subset=data_type,
            feature=("raw" if data_transform == "raw" else "spectrogram"),
            model="DAMUSIC",
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            array_aug=array_aug,
            transform_when_train=transform_when_train,
        )

    def _build_items(self):
        self.items = list(range(len(self.data_arr)))

    def _load_audio_np(self, item_idx: int) -> np.ndarray:
        item = self.data_arr[item_idx]
        audio = load_numpy(os.path.join(self.data_dir, item["raw"]))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio

    def _load_doas(self, item_idx: int) -> torch.Tensor:
        item = self.data_arr[item_idx]
        loc_theta = item["loc_azimuth"]
        # original: sv,doas = array_aug(..., loc_theta, ...)
        # keep doas as scalar
        return torch.tensor(np.array([loc_theta], dtype=np.float32))


class AV16_Dataset(BaseAudioDataset):
    def __init__(
        self,
        processed_root: str,
        mic_offsets: np.ndarray,
        subset: str = "train",
        data_transform: str = "stft",
        noise_aug: bool = False,
        resize_hw: Tuple[int, int] = (257, 64),
        cam: int = 1,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        num_percent: float = 1.0,
    ):
        self.root = Path(processed_root)
        self.cam = int(cam)
        self.num_percent = float(num_percent)

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        transform_when_train = True

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.5, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=True, resize_hw=resize_hw, mode="magphase"))

        array_aug = ArrayAugmentor(
            mic_offsets,
            ArrayAugConfig(enabled=True, interval=None, mic_center=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        )

        super().__init__(
            subset=subset,
            feature=("raw" if data_transform == "raw" else feature),
            model=model,
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            array_aug=array_aug,
            transform_when_train=transform_when_train,
        )

    def _build_items(self):
        items: List[Tuple[Path, Path, str]] = []
        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            take = int(len(audio_files) * self.num_percent)
            for ap in audio_files[:take]:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    items.append((ap, gp, seq_dir.name))

        if len(items) == 0:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        self.items = items
        print(f"[OK] AV16 indexed {len(self.items)} samples from {self.root}")

    def _load_audio_np(self, item: Tuple[Path, Path, str]) -> np.ndarray:
        ap, _, _ = item
        audio = np.load(str(ap)).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio

    def _load_doas(self, item: Tuple[Path, Path, str]) -> torch.Tensor:
        _, gp, _ = item
        gt = load_gt_dict(gp)
        xyz = np.asarray(gt["gt3d_xyz"], dtype=np.float32)
        loc_theta_deg = np.array(doa_xy_deg_from_xyz(xyz), dtype=np.float32)
        return torch.tensor(loc_theta_deg, dtype=torch.float32)

class AV16_Dataset_pretrain(BasePretrainDataset):
    def __init__(
        self,
        processed_root: str,
        subset: str = "train",
        data_transform: str = "stft",
        noise_aug: bool = False,
        resize_hw: Tuple[int, int] = (257, 64),
        cam: int = 1,
        model: str = "DAMUSIC",
        feature: str = "spectrogram",
        num_percent: float = 1.0,
    ):
        self.root = Path(processed_root)
        self.cam = int(cam)
        self.num_percent = float(num_percent)

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        noise_cfg = NoiseAugConfig(enabled=noise_aug, p=0.4, snr_db=None, snr_range=(-10, 30))
        spec_extractor = SpectrogramExtractor(SpecConfig(return_corr=False, resize_hw=resize_hw, mode="magphase"))
        masker = MicMasker(MaskConfig(enabled=True, p=1.0, ratio=0.1, patch_size=10))

        super().__init__(
            subset=subset,
            feature=("raw" if data_transform == "raw" else feature),
            noise_cfg=noise_cfg,
            spec_extractor=spec_extractor,
            masker=masker,
            model=model,
        )

    def _build_items(self):
        items: List[Tuple[Path, Path, str]] = []
        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            take = int(0.8 * len(audio_files) * self.num_percent)  # match your original pretrain: first 80%
            for ap in audio_files[:take]:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    items.append((ap, gp, seq_dir.name))

        if len(items) == 0:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        self.items = items
        print(f"[OK] AV16-pretrain indexed {len(self.items)} samples from {self.root}")

    def _load_audio_np(self, item: Tuple[Path, Path, str]) -> np.ndarray:
        ap, _, _ = item
        audio = np.load(str(ap)).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio


def quick_inspect_batch(batch):
    """
    Compatible with:
      1) spec, doas, sv, corr
      2) audio, doas, sv
      3) pretrain: spec_masked, spec_ori, mask
    """
    if len(batch) == 4:
        x, doas, sv, corr = batch
        print("[BATCH] spectrogram:", tuple(x.shape), x.dtype)
        print("[BATCH] correlation:", tuple(corr.shape), corr.dtype)
        print("[BATCH] doas:", tuple(doas.shape), doas.dtype,
              "min/max:", float(doas.min()), float(doas.max()))
        print("[BATCH] sv:", tuple(sv.shape), sv.dtype)

    elif len(batch) == 3:
        x0, x1, x2 = batch
        # heuristics: pretrain returns (spec_masked, spec_ori, mask)
        if torch.is_tensor(x2) and x2.ndim == 3:
            spec_masked, spec_ori, mask = x0, x1, x2
            print("[BATCH] pretrain spec_masked:", tuple(spec_masked.shape), spec_masked.dtype)
            print("[BATCH] pretrain spec_ori:", tuple(spec_ori.shape), spec_ori.dtype)
            print("[BATCH] pretrain mask:", tuple(mask.shape), mask.dtype,
                  "mask ratio:", float(mask.mean()))
        else:
            x, doas, sv = batch
            print("[BATCH] audio/raw:", tuple(x.shape), x.dtype)
            print("[BATCH] doas:", tuple(doas.shape), doas.dtype,
                  "min/max:", float(doas.min()), float(doas.max()))
            print("[BATCH] sv:", tuple(sv.shape), sv.dtype)
    else:
        raise ValueError("Unexpected batch length:", len(batch))
