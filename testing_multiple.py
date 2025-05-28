import os

from matplotlib import pyplot as plt
from torch import nn
import librosa
import sys
from scipy.signal import find_peaks
sys.path.append('../')
from model.DeepMusic_auto import DeepMusic_plus_class as DeepMusic_plus_new
from dataset.data_loader_new import DAMUSIC_Loader
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N, T, M = 4,1600,1
run_dataset = 0
dataset_zoo = ['simulation','soclas','afpild','rsl']
snr = None
noise_aug = False
if run_dataset==0:
    mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
    [ -45.7/1000/2,  45.7/1000/2, 0.0],
    [-45.7/1000/2,  -45.7/1000/2, 0.0],
    [45.7/1000/2,  -45.7/1000/2, 0.0]])
    data_dir = '/media/kemove/T9/sound_source_loc/simulation_data'
    val_dataset   = DAMUSIC_Loader(root=data_dir,mic_offsets=mic_offset,subset="val",model='CNN',geometry_aug=False,noise_aug=noise_aug,time_aug=False,coherent=1,num_source=M,snr=snr)
    lr = 0.001

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
target_noise_list = [snr]
for j in [1.0]:
    for k in range(1,2):
        print(f"testing {dataset_zoo[run_dataset]} on {j} percent data with pretrain = {k}")
        model = DeepMusic_plus_new(N, T, M,device=device).to(device)
        model.load_state_dict((torch.load(f'/media/kemove/T9/sound_source_loc/simulation_data/5-5/saved_simulation_attention/ours_{j}_{k}_multiple/best_model')))
        # model.load_state_dict(torch.load(f'/media/kemove/T9/sound_source_loc/simulation_data/3-31/saved_{dataset_zoo[run_dataset]}_attention_new/ours_{j}_{int(k)}_{M}/best_model'))
        model.eval()
        for target_noise in target_noise_list:
            save_dir = f"/media/kemove/T9/sound_source_loc/result/{dataset_zoo[run_dataset]}/unknown"
            os.makedirs(save_dir, exist_ok=True)
            predictions = []
            ground_truths = []
            gt_class = []
            pred_clss = []
            for i, (test_sample, gt, sv, correlation) in enumerate(tqdm(val_loader)):
                test_sample, sv, correlation = test_sample.to(device), sv.to(device), correlation.to(device)

                with torch.no_grad():
                    inference = model(test_sample, sv, correlation)
                    output = inference[1].squeeze(0).cpu().numpy()
                    peaks, _ = find_peaks(output, height=0.1, distance=5)
                    predicted_source = inference[2].squeeze(0).cpu().numpy()
                    # peaks, _ = find_peaks(output, height=0.01, distance=5)
                    K = M
                    topk_indices = peaks[np.argsort(output[peaks])[-K:]]
                    if len(topk_indices) < K:
                        max_index = np.argmax(output)
                        topk_indices = np.array([max_index] * K)
                    topk_indices = np.sort(topk_indices)

                    predictions.append(topk_indices)
                    ground_truths.append(gt.squeeze(0).cpu().numpy())
                    gt_class.append([M])
                    pred_clss.append(np.argmax(predicted_source))
                    # print(np.argmax(predicted_source))
                    acc = accuracy_score(gt_class, pred_clss)

                    # print(topk_indices,gt)
                    # predictions.append(np.argmax(output,axis=0))
                    # ground_truths.append(gt.item())
                    predictions_array = np.array(predictions) % 360
                    ground_truths_array = np.array(ground_truths) % 360
                    optimized_mae_loss = np.mean(np.abs((predictions_array - ground_truths_array + 180) % 360 - 180))
                    # print(topk_indices,gt.squeeze(0).cpu().numpy(),optimized_mae_loss,acc)

                del test_sample, sv, correlation, output

                if i % 100 == 0:
                    torch.cuda.empty_cache()

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)

            np.save(os.path.join(save_dir, f"predictions_ours_{target_noise}_{int(j*10)}_{k}_{M}.npy"), predictions)
            # np.save(os.path.join(save_dir, f"predictions_ours_{target_noise}_5.npy"), predictions)
            np.save(os.path.join(save_dir, f"ground_truths_un_{M}.npy"), ground_truths)
            print(f"The loss is {optimized_mae_loss}, the cla acc is {acc}")
            print(f"Predictions saved in '{save_dir}' directory.")
