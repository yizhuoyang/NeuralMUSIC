import os
import pyroomacoustics as pra
import warnings
import random
import numpy as np
from random import uniform, sample
from tqdm import tqdm
from pyroomacoustics import doa, Room, ShoeBox
warnings.filterwarnings("ignore")

def rotate_array(mic_locs, rotation_angle_deg,mic_center):
    """
    Rotates the entire microphone array around its center by a given angle.

    :param mic_locs: (3, num_mics) array of microphone positions
    :param rotation_angle_deg: Rotation angle in degrees (randomly chosen)
    :return: Rotated microphone positions
    """
    rotation_angle_rad = np.radians(rotation_angle_deg)
    rotation_matrix = np.array([
        [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad), 0],
        [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad),  0],
        [0, 0, 1]  # No rotation in the z-axis
    ])
    # Rotate each microphone location around the center
    rotated_mic_locs = rotation_matrix @ (mic_locs - mic_center) + mic_center
    return rotated_mic_locs


def data_generation(num_source,dataset,fs = 16000,max_order_ub = 4,r_lb=0.5,r_ub =2,train=True,save_dir='./data',cohenet=False):
    if train:
        generate_sample = 100
    else:
        generate_sample = 25

    doas_deg = np.linspace(start=0, stop=359, num=360, endpoint=True)
    mic_center= np.c_[[3,3,1]]
    mic_locs = mic_center + np.c_[  [ 45.7/1000/2, 45.7/1000/2, 0.0],
                [ -45.7/1000/2,  45.7/1000/2, 0.0],
                [-45.7/1000/2,  -45.7/1000/2, 0.0],
                [45.7/1000/2,  -45.7/1000/2, 0.0]]

    save_dir = os.path.join(save_dir,f"NS_{num_source}")
    os.makedirs(save_dir,exist_ok=True)

    for k in tqdm(range(generate_sample), desc="Generating Samples", position=0, leave=True):
        for i, doa_deg in enumerate(doas_deg):

            doa_list = [doa_deg]
            r = uniform(r_lb,r_ub)
            max_order = random.randint(0,max_order_ub)
            doa_rad = np.deg2rad(doa_deg)
            room_dim = [7,7,3]
            if cohenet:
                room = ShoeBox(room_dim, fs=fs, max_order=max_order)
            else:
                room = ShoeBox(room_dim, fs=fs)

            room.add_microphone_array(mic_locs)
            for j in range(num_source):
                if j==0:
                    source_loc = mic_center[:,0] + np.c_[r*np.cos(doa_rad), r*np.sin(doa_rad), 0][0]
                else:
                    doa_new = random.randint(0,359)
                    doa_list.append(doa_new)
                    doa_new = np.deg2rad(doa_new)
                    source_loc = mic_center[:,0] + np.c_[r*np.cos(doa_new), r*np.sin(doa_new), 0][0]

                sampled_audio = dataset.__getitem__(random.randint(0,dataset.__len__()-1)).data
                room.add_source(source_loc, signal=sampled_audio)

            room.simulate(snr=30)
            signals = room.mic_array.signals
            if num_source==1:
                np.save(os.path.join(save_dir,f"degree_{doa_list[0]}__times{k}"),signals)
            elif num_source==2:
                np.save(os.path.join(save_dir,f"degree_{doa_list[0]}-{doa_list[1]}_times{k}"),signals)
            elif num_source==3:
                np.save(os.path.join(save_dir,f"degree_{doa_list[0]}-{doa_list[1]}-{doa_list[2]}_times{k}"),signals)

if __name__ == "__main__":
    dataset = pra.datasets.GoogleSpeechCommands(basedir='/examples/google_speech_commands',download=True)
    selected_word = 'yes'
    matches = dataset.filter(word=selected_word)
    print(matches)
    print("Number of '%s' samples : %d" % (selected_word, len(matches)))
    for num_source in range(1,2):
        print(f"Generating source {num_source}")
        data_generation(num_source=num_source,dataset=dataset,train=True,cohenet=True, save_dir='../simulation_data/train/coherent/')
