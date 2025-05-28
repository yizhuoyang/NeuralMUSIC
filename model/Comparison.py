import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
import sys
import torch.nn.functional as F
sys.path.append("../")
from DeepMucis_plus.utlis.util import SteeringVector, ModeVector_torch,Grid


class Encoder_cls(nn.Module):
    def __init__(self, dropout_prob=0.1):  # Add dropout_prob to control
        super(Encoder_cls, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)#16
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) #32
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(skip1)
        skip2 = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(skip2)
        skip3 = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(skip3)
        skip4 = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(skip4)
        return x

class DeepCNN(nn.Module):
    def __init__(self,grid_size,input_channel=8,dropout_rate=0.3):
        super(DeepCNN, self).__init__()
        self.grid_size = grid_size
        self.DropOut = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(64)
        self.BatchNorm2 = nn.BatchNorm2d(128)
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.BatchNorm4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(574592, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.grid_size)
        self.ReLU = nn.ReLU()

    def forward(self, X):
        B, C, _, _ = X.size()  # Get batch size

        X = self.conv1(X)
        # X = self.DropOut(X)
        X = self.BatchNorm1(X)
        X = self.ReLU(X)

        X = self.conv2(X)
        # X = self.DropOut(X)
        X = self.BatchNorm2(X)
        X = self.ReLU(X)

        X = self.conv3(X)
        # X = self.DropOut(X)
        X = self.BatchNorm3(X)
        X = self.ReLU(X)

        X = self.conv4(X)
        # X = self.DropOut(X)
        X = self.BatchNorm4(X)
        X = self.ReLU(X)

        X = X.view(B, -1)
        X = self.ReLU(self.fc1(X))
        X = self.ReLU(self.fc2(X))
        X = self.fc3(X)
        return X

class DeepMusic(nn.Module):
    def __init__(self,grid_size=360,input_channel=12,dropout_rate=0.6,use_sigmoid=True):
        super(DeepMusic, self).__init__()
        # CNN Layers
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(input_channel,256, kernel_size=5)
        self.conv2 = nn.Conv2d(256,256, kernel_size=5)
        self.conv3 = nn.Conv2d(256,256, kernel_size=3)
        self.conv4 = nn.Conv2d(256,256, kernel_size=3)
        # Fully Connected Layers
        self.DropOut = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(692224, 1024)
        self.fc2 = nn.Linear(1024, self.grid_size)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self._initialize_weights()
        self.BatchNorm1 = nn.BatchNorm2d(256)
        self.BatchNorm2 = nn.BatchNorm2d(256)
        self.BatchNorm3 = nn.BatchNorm2d(256)
        self.BatchNorm4 = nn.BatchNorm2d(256)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        B, C, _, _ = X.size()

        X = self.conv1(X)
        X = self.DropOut(X)
        # X = self.BatchNorm1(X)
        X = self.ReLU(X)

        X = self.conv2(X)
        X = self.DropOut(X)
        # X = self.BatchNorm2(X)
        X = self.ReLU(X)

        X = self.conv3(X)
        X = self.DropOut(X)
        # X = self.BatchNorm3(X)
        X = self.ReLU(X)

        X = self.conv4(X)
        X = self.DropOut(X)
        # X = self.BatchNorm1(X)
        X = self.ReLU(X)

        X = X.view(B, -1)
        X = self.ReLU(self.fc1(X))
        if self.use_sigmoid:
            X = self.sigmoid(self.fc2(X))
        else:
            X = (self.fc2(X))
            X = X / X.max(dim=-1, keepdim=True)[0]
        return X,X



class DOAnetSPS(nn.Module):
    def __init__(self, sps_output_dim=360,dropout_rate=0.2):
        super(DOAnetSPS, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1),  # (B, 64, 257, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 64, 128, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 128, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 128, 64, 64)

            nn.Conv2d(128,256, kernel_size=3, padding=1),  # (B, 256, 64, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 256, 32, 64)
        )
        self.gru_hidden = 128        #128
        self.gru_layers = 2
        self.gru = nn.GRU(
            input_size=256 * 32,         #256
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_sps = nn.Linear(self.gru_hidden * 2, sps_output_dim)  # 双向GRU输出

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, -1)
        x, _ = self.gru(x)
        x = x[:,0,:]
        sps_out = self.fc_sps(x)  # (Batch, sps_output_dim)

        return x,sps_out

class DOAnetSPS_class(nn.Module):
    def __init__(self, sps_output_dim=360,dropout_rate=0.2):
        super(DOAnetSPS_class, self).__init__()
        self.relu    =  nn.LeakyReLU(0.1)
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1),  # (B, 64, 257, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 64, 128, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 128, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 128, 64, 64)

            nn.Conv2d(128,256, kernel_size=3, padding=1),  # (B, 256, 64, 64)
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 256, 32, 64)
        )

        self.gru_hidden = 128        #128
        self.gru_layers = 2
        self.gru = nn.GRU(
            input_size=256 * 32,         #256
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_sps = nn.Linear(self.gru_hidden * 2, sps_output_dim)  # 双向GRU输出
        self.source_prediction1 = nn.Linear(self.gru_hidden * 2,128)
        self.source_prediction2 = nn.Linear(128,4)
    def forward(self, x):

        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, -1)
        x, _ = self.gru(x)
        x = x[:,0,:]
        sps_out = self.fc_sps(x)  # (Batch, sps_output_dim)
        num_source = self.relu(self.source_prediction1(x))
        num_source = self.source_prediction2(num_source)
        return x,sps_out,num_source

class Encoder(nn.Module):
    def __init__(self, dropout_prob=0.1):  # Add dropout_prob to control
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)#16
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #32
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 257, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(257)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout2D for feature maps
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        # skip1 = self.dropout(skip1)  # Dropout after activation
        x = self.pool1(skip1)

        skip2 = F.relu(self.bn2(self.conv2(x)))
        # skip2 = self.dropout(skip2)
        x = self.pool2(skip2)

        skip3 = F.relu(self.bn3(self.conv3(x)))
        # skip3 = self.dropout(skip3)
        x = self.pool3(skip3)

        skip4 = F.relu(self.bn4(self.conv4(x)))
        # skip4 = self.dropout(skip4)
        x = self.pool4(skip4)

        return x, [skip1, skip2, skip3, skip4]

class DAMUSIC(nn.Module):
    def __init__(self, N, T, M, mic_positions,device='cuda'):
        super(DAMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi , np.pi, 360).to(device)
        self.device = device
        self.input_size = self.N
        self.hidden_size = 2 * self.N
        self.encoder = Encoder()
        self.relu    =  nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(16,32)
        self.dropout = nn.Dropout2d(p=0.3)
        self.fc_spec1 = nn.Linear(360,128)
        self.fc_spec2 = nn.Linear(128,self.M)
        self.convs2  = nn.Conv1d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.convs1  = nn.Conv1d(in_channels=257, out_channels=1, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.fc1 = nn.Linear(self.angels.shape[0]*257, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.M)
        grid = Grid()
        steer_vector_calc = ModeVector_torch(torch.tensor(mic_positions.T), 16000, 512, 343, grid, "far", precompute=True)
        self.sv = steer_vector_calc.mode_vec

    def spectrum_calculation(self, Un):
        sv = self.sv
        batch_size, F, M, _ = Un.shape
        Un_UnH = torch.matmul(Un, torch.conj(Un).transpose(-2, -1))
        sv = sv.to(dtype=Un.dtype, device=Un.device)
        # bs,_, num_angles, _ = sv.shape
        sv = sv.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, F, N_angles, M]
        sv_H = torch.conj(sv).transpose(-2, -1)  # [batch_size, F, M, N_angles]
        denominator_matrix = torch.matmul(sv_H, torch.matmul(Un_UnH, sv))
        spectrum_eq = denominator_matrix.diagonal(dim1=-2, dim2=-1).real
        spectrum_eq = spectrum_eq.clamp(min=1e-6)
        spectrum = 1 / spectrum_eq
        return spectrum

    def pre_MUSIC(self,Rz):
        Rz = (Rz + Rz.transpose(-1, -2).conj()) / 2
        Rz = Rz + 1e-4 * torch.eye(Rz.shape[-1], device=Rz.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un = eigenvectors[:, :, :, self.M:]
        return self.spectrum_calculation(Un)

    def forward(self, X):
        self.BATCH_SIZE = X.shape[0]
        denoised_spec = X
        x,_ = self.encoder(denoised_spec)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1], 4, 8)
        Rx_real, Rx_imag = torch.chunk(x, chunks=2, dim=-1)
        feature = torch.complex(Rx_real, Rx_imag)
        spectrum  = self.pre_MUSIC(feature)
        spectrum = spectrum.view(self.BATCH_SIZE,-1)
        DOA = self.relu(self.fc1(spectrum))
        DOA = self.relu((self.fc2(DOA)))
        DOA = self.fc3(DOA)
        return DOA

if __name__ == "__main__":
    batch_size = 4
    freq_bins = 257
    time_frames = 64
    channels = 12
    dummy_input = torch.randn(batch_size,  channels,freq_bins, time_frames)  # (B, 257, 64, 8)
    model = DOAnetSPS(sps_output_dim=360)
    sps_output = model(dummy_input)

    print(f"SPS output shape: {sps_output.shape}")  # 应为 (4, 360)
