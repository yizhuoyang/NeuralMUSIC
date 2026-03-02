# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append("../")
from model.NeuralMusic import NeuralMusic

from dataset.data_loader import (
    DAMUSIC_Loader, SoClas_database, AFPILD_raw_Dataset, RSL_database,
    AV16_Dataset, CAV3D_Dataset
)

N, T = 4, 1600
M = 2                    
RUN_DATASET = 4            
PREDICT_NUM_SOURCES = False 
ATTENTION = False
SNR = None
NOISE_AUG = False
MODE = "all"              

CKPT_PATH = "/home/Disk/yyz/deepmusic++/New_exp/saved_av16_attention_new/ours_1.0_1_2/best_model"
SAVE_DIR = "/home/Disk/yyz/deepmusic++/result/" 

BATCH_SIZE = 1
NUM_WORKERS = 4

PEAK_HEIGHT = 0.1
PEAK_DISTANCE = 5

EMPTY_CACHE_EVERY = 0  


dataset_zoo = ["simulation", "soclas", "afpild", "av16"]



def circular_abs_deg(a, b):
    """a,b in degrees; returns |wrap(a-b)| in [0,180]. broadcastable."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def pick_topk_indices(prob_1d, K, peak_height=0.1, peak_distance=5):
    peaks, _ = find_peaks(prob_1d, height=peak_height, distance=peak_distance)

    if peaks.size >= K:
        topk = peaks[np.argsort(prob_1d[peaks])[-K:]]
        return np.sort(topk)
    topk = np.argsort(prob_1d)[-K:]
    return np.sort(topk)


def compute_mae_deg(preds, gts):
    preds = np.asarray(preds) % 360
    gts = np.asarray(gts) % 360
    return float(np.mean(circular_abs_deg(preds, gts)))


# -------------------------
# Dataset builder
# -------------------------
def build_val_dataset(run_dataset, M, snr=None, noise_aug=False, mode="all"):
    input_channel = 8

    if run_dataset == 0:  # simulation
        mic_offset = np.array([
            [ 45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2, -45.7/1000/2, 0.0],
            [ 45.7/1000/2, -45.7/1000/2, 0.0],
        ])
        data_dir = "/media/kemove/T9/sound_source_loc/simulation_data"
        val_dataset = DAMUSIC_Loader(
            root=data_dir, mic_offsets=mic_offset,
            subset="val", model="CNN",
            geometry_aug=False, noise_aug=noise_aug, time_aug=False,
            coherent=1, num_source=M, snr=snr, mode=mode
        )
        return val_dataset, input_channel

    if run_dataset == 1:  # soclas
        mic_offset = np.array([
            [-45.7/1000/2,  45.7/1000/2, 0.0],
            [ 45.7/1000/2,  45.7/1000/2, 0.0],
            [ 45.7/1000/2, -45.7/1000/2, 0.0],
            [-45.7/1000/2, -45.7/1000/2, 0.0],
        ])
        data_dir = "/media/kemove/T9/sound_source_loc/SoClas_database/SoClas_database/Segmented_Sound"
        val_dataset = SoClas_database(root=data_dir, mic_offsets=mic_offset, subset="val")
        return val_dataset, input_channel

    if run_dataset == 2:  # afpild
        mic_offset = np.array([
            [ 45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2,  45.7/1000/2, 0.0],
            [-45.7/1000/2, -45.7/1000/2, 0.0],
            [ 45.7/1000/2, -45.7/1000/2, 0.0],
        ])
        data_dir = "/media/kemove/T9/sound_source_loc/afpild_data"
        val_dataset = AFPILD_raw_Dataset(dataset_dir=data_dir, mic_offsets=mic_offset, data_type="test")
        return val_dataset, input_channel

    if run_dataset == 3:  # av16
        input_channel = 32
        mic_offset = np.array([
            [-0.10000,  0.40000,  0.0],
            [-0.07071,  0.32929,  0.0],
            [ 0.00000,  0.30000,  0.0],
            [ 0.07071,  0.32929,  0.0],
            [ 0.10000,  0.40000,  0.0],
            [ 0.07071,  0.47071,  0.0],
            [ 0.00000,  0.50000,  0.0],
            [-0.07071,  0.47071,  0.0],

            [-0.10000, -0.40000,  0.0],
            [-0.07071, -0.47071,  0.0],
            [ 0.00000, -0.50000,  0.0],
            [ 0.07071, -0.47071,  0.0],
            [ 0.10000, -0.40000,  0.0],
            [ 0.07071, -0.32929,  0.0],
            [ 0.00000, -0.30000,  0.0],
            [-0.07071, -0.32929,  0.0],
        ])
        data_dir_val = f"/media/kemove/T9/sound_source_loc/AV1_processed/{M}p/test"
        val_dataset = AV16_Dataset(data_dir_val, mic_offsets=mic_offset, noise_aug=False, subset="test")
        return val_dataset, input_channel
    
    raise ValueError(f"Unknown RUN_DATASET={run_dataset}")


# -------------------------
# Main
# -------------------------
def main():
    dataset_name = dataset_zoo[RUN_DATASET]
    print(f"[Config] dataset={dataset_name}  M={M}  predict_num_sources={PREDICT_NUM_SOURCES}  attention={ATTENTION}")
    print(f"[Config] ckpt={CKPT_PATH}")

    # dataset
    val_dataset, input_channel = build_val_dataset(RUN_DATASET, M=M, snr=SNR, noise_aug=NOISE_AUG, mode=MODE)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = NeuralMusic(
        N=N, T=T, M=M, device=device,
        attention=ATTENTION,
        input_channel=input_channel,
        predict_num_sources=PREDICT_NUM_SOURCES,
        max_sources=N,
    ).to(device)

    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    save_dir = os.path.join(SAVE_DIR, dataset_name, "unknown" if PREDICT_NUM_SOURCES else "")
    os.makedirs(save_dir, exist_ok=True)

    predictions, ground_truths, probs = [], [], []
    pred_classes, gt_classes = [], []

    with torch.inference_mode():
        for i, (test_sample, gt, sv, correlation) in enumerate(tqdm(val_loader)):
            test_sample = test_sample.to(device, non_blocking=True)
            sv = sv.to(device, non_blocking=True)
            correlation = correlation.to(device, non_blocking=True)

            out = model(test_sample, sv, correlation)

            spectrum = out[1].squeeze(0).detach().cpu().numpy()  # (360,)

            topk = pick_topk_indices(
                spectrum, K=M,
                peak_height=PEAK_HEIGHT,
                peak_distance=PEAK_DISTANCE,
            )

            predictions.append(topk)
            ground_truths.append(gt.squeeze(0).cpu().numpy())
            probs.append(spectrum)

            if PREDICT_NUM_SOURCES:
                ns = out[2].squeeze(0).detach().cpu().numpy()
                pred_classes.append(int(np.argmax(ns)))
                gt_classes.append(int(M)) 

            if EMPTY_CACHE_EVERY > 0 and (i % EMPTY_CACHE_EVERY == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = np.asarray(predictions) % 360
    ground_truths = np.asarray(ground_truths) % 360
    probs = np.asarray(probs)

    mae = compute_mae_deg(predictions, ground_truths)
    print(f"[Done] MAE(deg) = {mae:.4f}")

    # save
    np.save(os.path.join(save_dir, f"predictions_ours_M{M}.npy"), predictions)
    np.save(os.path.join(save_dir, f"probs_ours_M{M}.npy"), probs)
    np.save(os.path.join(save_dir, f"ground_truths_ours_M{M}.npy"), ground_truths)

    if PREDICT_NUM_SOURCES:
        pred_classes = np.asarray(pred_classes)
        gt_classes = np.asarray(gt_classes)
        acc = float((pred_classes == gt_classes).mean())
        np.save(os.path.join(save_dir, f"pred_class_M{M}.npy"), pred_classes)
        np.save(os.path.join(save_dir, f"gt_class_M{M}.npy"), gt_classes)
        print(f"[Done] Class Acc = {acc:.4f}")

    print(f"[Saved] -> {save_dir}")


if __name__ == "__main__":
    main()