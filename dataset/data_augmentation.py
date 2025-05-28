import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(audiodata, target_snr_db):
    """
    Add Gaussian noise to a multi-channel audio signal at a specified SNR level.

    :param audiodata: Original multi-channel audio (shape: (num_mics, num_samples))
    :param target_snr_db: Target SNR level in dB (same for all mics)
    :return: Noisy audio with the specified SNR
    """
    num_mics, num_samples = audiodata.shape
    noisy_audio = np.zeros_like(audiodata)

    for i in range(num_mics):
        signal = audiodata[i]  # Extract one mic channel
        noise = np.random.normal(0, 1, num_samples)  # Generate Gaussian noise

        # Compute power of signal and noise
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        # Compute noise scaling factor to match desired SNR
        snr_scaling_factor = np.sqrt(signal_power / (10 ** (target_snr_db / 10) * noise_power))

        # Scale and add noise
        noise = noise * snr_scaling_factor
        noisy_audio[i] = signal + noise  # Add noise to original signal

    return noisy_audio


if __name__ == "__main__":
    ## The function for SNR simulation
    fs = 16000  # Sampling rate
    audiodata = np.load('/media/kemove/T9/sound_source_loc/simulation_data/train/coherent/NS_1/degree_0.0__times0.npy')  # Simulated audio data (4 mics, 1 sec)
    target_snr_db = -10  # Set SNR level
    noisy_audio = add_gaussian_noise(audiodata, target_snr_db)
    mic_channel = 0
    original_signal = audiodata[mic_channel]
    noisy_signal = noisy_audio[mic_channel]

    # Plot the original and noisy waveforms
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(original_signal, label="Original Signal", color="blue")
    plt.title("Original Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(noisy_signal, label=f"Noisy Signal (SNR={target_snr_db} dB)", color="red")
    plt.title("Noisy Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

