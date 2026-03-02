import numpy as np
import torch
import torch.nn as nn
import warnings
import time
import sys

from torch.nn import MultiheadAttention

sys.path.append("../")
from utlis.util import SteeringVector, ModeVector_torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channel=8,dropout_prob=0.3):  # Add dropout_prob to control
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)#16
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

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        # skip1 = self.dropout(skip1)
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

class Encoder_cls(nn.Module):
    def __init__(self, dropout_prob=0.1,input_channel=8):  # Add dropout_prob to control
        super(Encoder_cls, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, padding=1)#16
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

        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout2D for feature maps

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        # skip1 = self.dropout(skip1)
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

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(257, 128, kernel_size=2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(128)  # Batch Norm
        self.conv1 = nn.Conv2d(128 + 257, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)  # Batch Norm

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn_up4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x, skips):
        x = self.bn_up1(self.up1(x))
        x = torch.cat([x, skips[3]], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.bn_up2(self.up2(x))
        x = torch.cat([x, skips[2]], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn_up3(self.up3(x))
        x = torch.cat([x, skips[1]], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.bn_up4(self.up4(x))
        x = F.interpolate(x, size=skips[0].shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skips[0]], dim=1)
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.final_conv(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

class Grid:
    def __init__(self):
        self.x = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_x.npy')
        self.y = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_y.npy')
        self.z = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_z.npy')
        
def soft_argmax_peak_refined(spectrum, alpha=20.0, top_k=5):
    batch_size, num_angles = spectrum.shape
    angles = torch.linspace(0, 360, spectrum.shape[-1], device=spectrum.device)
    top_indices = torch.topk(spectrum, k=top_k, dim=-1).indices
    top_spectrums = torch.gather(spectrum, dim=-1, index=top_indices)
    top_angles = torch.gather(angles.expand(batch_size, -1), dim=-1, index=top_indices)
    weights = torch.softmax(alpha * top_spectrums, dim=-1)
    doa_estimation = torch.sum(weights * top_angles, dim=-1)
    return doa_estimation


class DeepMusic_pretrain(nn.Module):
    def __init__(self,input_channel=8):
        super(DeepMusic_pretrain, self).__init__()
        self.encoder = Encoder(input_channel=input_channel)
        self.decoder = Decoder()
    def forward(self, x: torch.Tensor):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def hermitianize(R: torch.Tensor) -> torch.Tensor:
    # (..,M,M) complex
    return (R + R.transpose(-1, -2).conj()) / 2


# -----------------------------------------
# DeepMusic_plus (Unified)
# -----------------------------------------
class NeuralMusic(nn.Module):
    """
    Unified DeepMusic:
      - predict_source=False : fixed #sources = M_fixed (original DeepMusic_plus)
      - predict_source=True  : predict #sources (original DeepMusic_plus_class behavior)

    Forward I/O:
      X:  (B,C,F,T)
      sv: (B,F,A,M) or (B,F,M,A)  <-- both supported
      correlation: unused, kept for compatibility

    Returns:
      - predict_source=False : (DOA, spectrum)
      - predict_source=True  : (DOA, spectrum, num_source_logits)
        (you can also return M_hat if you want)
    """

    def __init__(
        self,
        N,
        T,
        M,
        device,
        attention=True,
        input_channel=8,
        predict_source=False,
        eps=1e-6,
        diag_eps=1e-5,
    ):
        super().__init__()
        self.N, self.T = N, T
        self.M_fixed = M
        self.device = device
        self.input_channel = int(input_channel)

        self.predict_source = bool(predict_source)
        self.attention = bool(attention)

        self.eps = float(eps)
        self.diag_eps = float(diag_eps)

        # ----- Encoders -----
        self.encoder = Encoder(input_channel=self.input_channel)

        if self.predict_source:
            self.encoder_class = Encoder_cls(input_channel=self.input_channel)
            self.source_prediction1 = nn.Linear(64 * 32, 128)
            self.source_prediction2 = nn.Linear(128, self.N)

        # ----- Rx head -----
        self.fc = nn.Linear(64, int(self.input_channel * self.input_channel / 2))

        # ----- Spectrum head -----
        self.convs2 = nn.Conv1d(257, 257, kernel_size=3, padding=1)
        self.convs1 = nn.Conv1d(257, 1, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # optional channel attention for (B,257,A)
        self.channel_attention = ChannelAttention(in_channels=257, reduction=16)

    # -------------------------
    # sv layout adapter
    # -------------------------
    @staticmethod
    def _ensure_sv_layout(sv: torch.Tensor, M: int) -> torch.Tensor:
        """
        Ensure sv is (B,F,A,M). Accepts:
          - (B,F,A,M) : ok
          - (B,F,M,A) : permute -> (B,F,A,M)
        """
        if sv.dim() != 4:
            raise ValueError(f"sv must be 4-D, got shape={tuple(sv.shape)}")

        if sv.shape[-1] == M:
            # (B,F,A,M)
            return sv
        if sv.shape[-2] == M:
            # (B,F,M,A) -> (B,F,A,M)
            return sv.permute(0, 1, 3, 2).contiguous()

        raise ValueError(f"Cannot infer sv layout with M={M}. sv shape={tuple(sv.shape)}")

    # -------------------------
    # spectrum computation
    # -------------------------
    @staticmethod
    def _spectrum_from_Un(Un: torch.Tensor, sv: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Un: (B,F,M,K) noise subspace
        sv: (B,F,A,M) or (B,F,M,A)
        return: (B,F,A)
        """
        B, F, M, K = Un.shape
        sv = NeuralMusic._ensure_sv_layout(sv, M).to(dtype=Un.dtype, device=Un.device)  # -> (B,F,A,M)

        # UnUnH: (B,F,M,M)
        UnUnH = Un @ Un.transpose(-2, -1).conj()

        # v = UnUnH @ sv (along M) -> (B,F,A,M)
        v = torch.einsum("bfmn,bfam->bfan", UnUnH, sv)

        # denom[a] = sv[a]^H v[a] -> (B,F,A)
        denom = torch.einsum("bfam,bfam->bfa", sv.conj(), v).real
        denom = denom.clamp_min(eps)
        return 1.0 / denom

    def _pre_music_fixedM(self, Rz: torch.Tensor, sv: torch.Tensor, M_use: int) -> torch.Tensor:
        """
        Rz: (B,F,M,M) complex
        sv: (B,F,A,M) or (B,F,M,A)
        """
        Rz = hermitianize(Rz)
        eye = torch.eye(Rz.size(-1), device=Rz.device, dtype=Rz.dtype).view(1, 1, Rz.size(-1), Rz.size(-1))
        Rz = Rz + self.diag_eps * eye

        _, evecs = torch.linalg.eigh(Rz)      # ascending
        evecs = torch.flip(evecs, dims=[-1])  # descending
        Un = evecs[..., M_use:]               # (B,F,M,M-M_use)
        return self._spectrum_from_Un(Un, sv, self.eps)

    def _pre_music_varM(self, Rz: torch.Tensor, sv: torch.Tensor, M_per_batch: torch.Tensor) -> torch.Tensor:
        """
        Variable M per sample, matches your original loop style.
        """
        Rz = hermitianize(Rz)
        eye = torch.eye(Rz.size(-1), device=Rz.device, dtype=Rz.dtype).view(1, 1, Rz.size(-1), Rz.size(-1))
        Rz = Rz + self.diag_eps * eye

        _, evecs = torch.linalg.eigh(Rz)
        evecs = torch.flip(evecs, dims=[-1])

        spectra = []
        B = Rz.size(0)
        for b in range(B):
            m_b = int(M_per_batch[b].item())
            Un_b = evecs[b, :, :, m_b:]  # (F,M,M-m_b)
            spec_b = self._spectrum_from_Un(Un_b.unsqueeze(0), sv[b:b+1], self.eps).squeeze(0)
            spectra.append(spec_b)
        return torch.stack(spectra, dim=0)  # (B,F,A)

    # -------------------------
    # Rx estimation
    # -------------------------
    def _estimate_Rz(self, X: torch.Tensor) -> torch.Tensor:
        """
        X -> encoder -> fc -> reshape -> complex Rz
        Output shape:
          z from encoder: (B,257,h,w)
          z after fc:     (B,257,input_channel^2/2)
          reshape:        (B,257,input_channel/2,input_channel)
          chunk:          (B,257,input_channel/2,input_channel/2) twice
        If input_channel == 2*M_mic, then Rz is (B,257,M_mic,M_mic)
        """
        z, _ = self.encoder(X)                        # (B,257,h,w)
        z = z.view(z.size(0), z.size(1), -1)          # (B,257,h*w)
        z = self.fc(z)                                # (B,257,input_channel^2/2)

        z = z.view(z.size(0), z.size(1), self.input_channel // 2, self.input_channel)
        Rx_real, Rx_imag = torch.chunk(z, chunks=2, dim=-1)  # split last dim
        Rz = torch.complex(Rx_real, Rx_imag)                 # (B,257,M,M) when input_channel=2M
        return Rz

    # -------------------------
    # Source-count prediction
    # -------------------------
    def _predict_sources(self, X: torch.Tensor):
        feat = self.encoder_class(X)                  # (B,32,?,?)
        feat = feat.view(feat.size(0), -1)            # -> (B, 64*32) (your original assumption)
        feat = self.relu(self.source_prediction1(feat))
        logits = self.source_prediction2(feat)        # (B,N)
        M_hat = torch.argmax(logits, dim=1)           # (B,)
        return logits, M_hat

    # -------------------------
    # forward
    # -------------------------
    def forward(self, X: torch.Tensor, sv: torch.Tensor, correlation=None):
        """
        Returns:
          - predict_source=False : DOA, spectrum
          - predict_source=True  : DOA, spectrum, num_source_logits
        """
        num_source_logits = None
        M_hat = None

        if self.predict_source:
            num_source_logits, M_hat = self._predict_sources(X)

        Rz = self._estimate_Rz(X)

        if self.predict_source:
            spectrum = self._pre_music_varM(Rz, sv, M_hat)   # (B,257,A)
        else:
            spectrum = self._pre_music_fixedM(Rz, sv, self.M_fixed)

        spectrum = self.relu(self.convs2(spectrum))          # (B,257,A)
        if self.attention:
            spectrum = self.channel_attention(spectrum)

        spectrum = self.sigmoid(self.convs1(spectrum)).squeeze(1)  # (B,A)
        DOA = soft_argmax_peak_refined(spectrum).unsqueeze(1)      # (B,1)

        if self.predict_source:
            # 如果你想额外返回 M_hat：return DOA, spectrum, num_source_logits, M_hat
            return DOA, spectrum, num_source_logits
        return DOA, spectrum