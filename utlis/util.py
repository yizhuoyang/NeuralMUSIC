import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
import torch
import torchaudio.transforms as T
import random
import pandas as pd

def downsample_audio(audio_tensor, target_length=1600):
    """
    Downsamples a multi-channel audio tensor to a fixed length (1600 samples).

    :param audio_tensor: Input tensor of shape (C, N), where N > 1600.
    :param target_length: The target number of samples (default = 1600).
    :return: Downsampled audio tensor of shape (C, 1600).
    """
    C, N = audio_tensor.shape  # Number of channels and original length

    # Define resampling transformation
    resampler = T.Resample(orig_freq=N, new_freq=target_length)

    # Apply resampling to all channels at once
    downsampled_audio = resampler(audio_tensor)  # Output shape: (C, 1600)

    return downsampled_audio

class ModeVector_torch:
    """
    A class for look-up tables of mode vectors using PyTorch.
    This look-up table is an outer product of three vectors running along candidate locations,
    time, and frequency. If the table is too large to store in memory, it computes values on the fly.
    """
    def __init__(self, L, fs, nfft, c, grid, mode="far", precompute=True):
        """
        Parameters
        ----------
        L: torch.Tensor
            Locations of the sensors (shape: [3, num_mics])
        fs: int
            Sampling frequency of the input signal
        nfft: int
            FFT length (must be even)
        c: float
            Speed of sound
        grid: object
            Grid object with attributes `x`, `y`, `z` (torch.Tensors)
        mode: str, optional
            'far' (default) or 'near', specifying the mode vector computation method
        precompute: bool, optional
            Whether to precompute the whole table (default: False)
        device: str, optional
            Device to store tensors (default: 'cpu')
        """
        if nfft % 2 == 1:
            raise ValueError("FFT length must be even.")

        self.precompute = precompute

        # Propagation vectors
        p_x = torch.tensor(grid.x).view(1, 1, -1)
        p_y = torch.tensor(grid.y).view(1, 1, -1)
        p_z = torch.tensor(grid.z).view(1, 1, -1)

        # Microphone locations
        r_x = torch.tensor(L[0]).view(1, -1, 1)
        r_y = torch.tensor(L[1]).view(1, -1, 1)
        r_z = torch.tensor(L[2]).view(1, -1, 1) if L.shape[0] == 3 else torch.zeros((1, L.shape[1], 1))

        if mode == "near":
            dist = torch.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)
        elif mode == "far":
            dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)
        else:
            raise ValueError("Mode must be 'near' or 'far'")
        self.tau = dist / c  # Time delay
        self.omega = 2 * torch.pi * fs * torch.arange(nfft // 2 + 1, ) / nfft
        if precompute:
            self.mode_vec = torch.exp(1j * self.omega[:, None, None] * self.tau)
        else:
            self.mode_vec = None

    def __getitem__(self, ref):
        """
        Retrieve mode vector values. Computes on the fly if not precomputed.
        """
        if self.precompute:
            return self.mode_vec[ref]

        w = self.omega[ref[0]].unsqueeze(-1) if isinstance(ref[0], slice) else self.omega[ref[0]]
        tau_selected = self.tau if len(ref) == 1 else self.tau[:, ref[1], :] if len(ref) == 2 else self.tau[:, ref[1], ref[2]]
        return torch.exp(1j * w * tau_selected)


class SteeringVector:
    def __init__(self, mic_positions, angles):
        """
        Compute the steering vector for an arbitrary microphone array shape, without considering frequency.

        :param mic_positions: Microphone coordinates, shape [M, 3] (M microphones with (x, y, z) positions).
        :param angles: List of incident angles, shape [N], in radians.
        """
        self.mic_positions = torch.tensor(mic_positions, dtype=torch.float32)  # [M, 3]
        self.angles = torch.tensor(angles, dtype=torch.float32)  # [N]

    def steering_vec(self):
        """
        Compute the steering vector for the microphone array, supporting any array shape and independent of frequency.

        Returns:
        --------
            torch.Tensor: Steering vector, shape [N, M] (N angles, M microphones).
        """
        # Compute the wave vector assuming the sound wave propagates in the XY plane.
        wave_vectors = torch.stack([
            torch.cos(self.angles),
            torch.sin(self.angles),
            torch.zeros_like(self.angles)  # Assume no variation in the Z direction.
        ], dim=-1)  # [N, 3]

        # Compute phase shifts (without frequency dependence)
        phase_shifts = torch.exp(-1j * torch.matmul(self.mic_positions.to('cuda'), wave_vectors.T))  # [M, N]

        return phase_shifts.T  # [N, M]


def filter_folders(folder_list, n):
    if n == 0:
        return folder_list
    return [folder for folder in folder_list if f"_{n}" in folder]

def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])

    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                # error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # error = ((prediction - targets + np.pi) % 2*np.pi) - np.
                error = ((prediction - targets + 180) % 360) - 180
                # Calculate RMSE over all permutations
                rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                # rmspe_val = torch.abs((1 / len(targets)) * torch.abs(error))
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            # Choose minimal error from all permutations
            rmspe_min = torch.min(rmspe_tensor)
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result

def normalize_magnitude(magnitude, method="min-max"):
    if method == "min-max":
        min_val = magnitude.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Min over time and frequency
        max_val = magnitude.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Max over time and frequency
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    elif method == "standard":
        mean = magnitude.mean()
        std = magnitude.std()
        magnitude = (magnitude - mean) / (std + 1e-8)
    return magnitude

def normalize_phase(phase, method="scale"):
    if method == "scale":
        phase = phase / torch.pi  # Scale phase to [-1, 1]
    elif method == "sincos":
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        return phase_sin, phase_cos  # Returns two tensors
    return phase



def compute_correlation_matrices_torch(X: torch.Tensor) -> torch.Tensor:
    X = X.permute(2, 1, 0) 
    C_hat = torch.matmul(X.unsqueeze(-1), X.unsqueeze(-2).conj())
    C_hat = C_hat.mean(dim=0)
    return C_hat

class Grid:
    def __init__(self):
        self.x = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_x.npy')
        self.y = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_y.npy')
        self.z = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_z.npy')

def array_aug(mic_offsets,locs,transform):
    if transform:
        random_integer = random.randint(0, 360)
    else:
        random_integer = 0
    grid = Grid()
    mic_center = np.array([[3, 3, 1]])
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


def load_dataframe(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_numpy(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        return None
    return data


if __name__ == "__main__":
    criterion = RMSPELoss()
    predictions = torch.tensor([100.0,20]).unsqueeze(0)
    targets = torch.tensor([350.0,140]).unsqueeze(0)
    loss = criterion(predictions, targets)
    print(loss)
