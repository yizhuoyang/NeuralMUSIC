from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from utlis.util import ModeVector_torch  # SteeringVector unused in your snippet


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def hermitianize(R: torch.Tensor) -> torch.Tensor:
    return 0.5 * (R + R.transpose(-1, -2).conj())

def soft_argmax_peak_refined(
    spectrum: torch.Tensor,
    alpha: float = 20.0,
    top_k: int = 5,
) -> torch.Tensor:
    """
    spectrum: (B, A)
    returns:  (B,) degrees in [0, 360)
    """
    B, A = spectrum.shape
    # Avoid duplicate 0/360 bin: generate A bins in [0,360)
    angles = torch.linspace(
        0.0, 360.0, A + 1,
        device=spectrum.device,
        dtype=spectrum.dtype
    )[:-1]

    k = min(top_k, A)
    top_idx = torch.topk(spectrum, k=k, dim=-1).indices                # (B,k)
    top_val = torch.gather(spectrum, dim=-1, index=top_idx)            # (B,k)
    top_ang = torch.gather(angles.expand(B, -1), dim=-1, index=top_idx)

    w = torch.softmax(alpha * top_val, dim=-1)
    return (w * top_ang).sum(dim=-1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward_feat(self, x: torch.Tensor) -> torch.Tensor:
        # feature BEFORE pooling (to match your original skip behavior)
        x = F.relu(self.bn(self.conv(x)), inplace=False)
        return self.drop(x)

    @staticmethod
    def pool2(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2, stride=2)

class Encoder(nn.Module):
    def __init__(self, input_channel: int = 8, dropout_prob: float = 0.0):
        super().__init__()
        self.b1 = ConvBlock2d(input_channel, 64, p_drop=dropout_prob)
        self.b2 = ConvBlock2d(64, 64, p_drop=dropout_prob)
        self.b3 = ConvBlock2d(64, 64, p_drop=dropout_prob)
        self.b4 = ConvBlock2d(64, 257, p_drop=dropout_prob)

    def forward(self, x: torch.Tensor):
        s1 = self.b1.forward_feat(x)
        x = self.b1.pool2(s1)

        s2 = self.b2.forward_feat(x)
        x = self.b2.pool2(s2)

        s3 = self.b3.forward_feat(x)
        x = self.b3.pool2(s3)

        s4 = self.b4.forward_feat(x)
        x = self.b4.pool2(s4)

        return x, [s1, s2, s3, s4]

class Encoder_cls(nn.Module):
    def __init__(self, input_channel: int = 8, dropout_prob: float = 0.0):
        super().__init__()
        self.b1 = ConvBlock2d(input_channel, 16, p_drop=dropout_prob)
        self.b2 = ConvBlock2d(16, 16, p_drop=dropout_prob)
        self.b3 = ConvBlock2d(16, 16, p_drop=dropout_prob)
        self.b4 = ConvBlock2d(16, 32, p_drop=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1.pool2(self.b1.forward_feat(x))
        x = self.b2.pool2(self.b2.forward_feat(x))
        x = self.b3.pool2(self.b3.forward_feat(x))
        x = self.b4.pool2(self.b4.forward_feat(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(257, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(128 + 257, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skips):
        x = self.up1(x)
        x = self.c1(torch.cat([x, skips[3]], dim=1))

        x = self.up2(x)
        x = self.c2(torch.cat([x, skips[2]], dim=1))

        x = self.up3(x)
        x = self.c3(torch.cat([x, skips[1]], dim=1))

        x = self.up4(x)
        x = F.interpolate(x, size=skips[0].shape[2:], mode="bilinear", align_corners=False)
        x = self.c4(torch.cat([x, skips[0]], dim=1))

        return self.final_conv(x)

class Autoencoder(nn.Module):
    def __init__(self, input_channel: int = 8):
        super().__init__()
        self.encoder = Encoder(input_channel=input_channel)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        z, skips = self.encoder(x)
        return self.decoder(z, skips)

class NeuralMusic_pretrain(nn.Module):
    def __init__(self, input_channel: int = 8):
        super().__init__()
        self.encoder = Encoder(input_channel=input_channel)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        z, skips = self.encoder(x)
        return self.decoder(z, skips)

class NeuralMusic(nn.Module):
    def __init__(
        self,
        N,
        T,
        M,
        device,
        attention: bool = True,
        input_channel: int = 8,
        eps: float = 1e-6,
        predict_num_sources: bool = False,
        max_sources: int | None = None,   # optional: override N for classification head output dim
    ):
        super().__init__()
        self.N, self.T = N, T
        self.M_fixed = M                 # fixed source count (if not predicting)
        self.input_channel = input_channel
        self.device = device
        self.eps = eps

        self.predict_num_sources = predict_num_sources
        self.max_sources = int(max_sources) if max_sources is not None else int(N)

        # Rx estimator encoder
        self.encoder = Encoder(input_channel=input_channel)

        # optional source-count classifier encoder
        if self.predict_num_sources:
            self.encoder_class = Encoder_cls(input_channel=input_channel)
            self.source_prediction1 = nn.Linear(64 * 32, 128)
            self.source_prediction2 = nn.Linear(128, self.max_sources)

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # After encoder output: (B,257,H,W) -> flatten HW -> FC -> (input_channel^2/2)
        self.fc = nn.Linear(64, int(input_channel * input_channel / 2))

        # spectrum head: (B,257,A) -> (B,257,A) -> (B,1,A)
        self.convs2 = nn.Conv1d(257, 257, kernel_size=3, padding=1)
        self.convs1 = nn.Conv1d(257, 1, kernel_size=3, padding=1)

        # attention option
        self.attention = attention
        self.channel_attention = ChannelAttention(in_channels=257, reduction=16)

    @staticmethod
    def _spectrum_from_Un(Un: torch.Tensor, sv: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Un: (B,F,M,M-d) noise subspace
        sv: (B,F,A,M)   steering vectors
        return: (B,F,A)
        """
        UnUnH = Un @ Un.transpose(-2, -1).conj()      # (B,F,M,M)

        sv = sv.to(device=Un.device, dtype=Un.dtype)  # (B,F,A,M)
        svH = sv.transpose(-2, -1).conj()             # (B,F,M,A)

        # denom[a] = sv[a]^H UnUnH sv[a]
        tmp = UnUnH @ sv.transpose(-2, -1).conj()     # (B,F,M,M)@(B,F,M,A) -> (B,F,M,A)
        denom = (svH * tmp).sum(dim=-2).real          # sum over M -> (B,F,A)

        denom = denom.clamp_min(eps)
        return 1.0 / denom

    def _pre_music_fixedM(self, Rz: torch.Tensor, sv: torch.Tensor, M_use: int) -> torch.Tensor:
        """
        Rz: (B,F,M,M) complex covariance
        sv: (B,F,A,M)
        M_use: fixed integer
        """
        Rz = hermitianize(Rz)
        eye = torch.eye(
            Rz.size(-1), device=Rz.device, dtype=Rz.dtype
        ).view(1, 1, Rz.size(-1), Rz.size(-1))
        Rz = Rz + 1e-5 * eye

        _, evecs = torch.linalg.eigh(Rz)
        evecs = torch.flip(evecs, dims=[-1])          # descending
        Un = evecs[..., M_use:]                       # (B,F,M,M-M_use)
        return self._spectrum_from_Un(Un, sv, self.eps)

    def _pre_music_varM(self, Rz: torch.Tensor, sv: torch.Tensor, M_per_batch: torch.Tensor) -> torch.Tensor:
        """
        Variable number of sources per sample (loop per batch).
        Rz: (B,F,M,M)
        sv: (B,F,A,M)
        M_per_batch: (B,)
        """
        Rz = hermitianize(Rz)
        eye = torch.eye(
            Rz.size(-1), device=Rz.device, dtype=Rz.dtype
        ).view(1, 1, Rz.size(-1), Rz.size(-1))
        Rz = Rz + 1e-5 * eye

        _, evecs = torch.linalg.eigh(Rz)
        evecs = torch.flip(evecs, dims=[-1])

        spectra = []
        B = Rz.size(0)
        for b in range(B):
            m_b = int(M_per_batch[b].item())
            Un_b = evecs[b, :, :, m_b:]               # (F,M,M-m_b)
            spec_b = self._spectrum_from_Un(
                Un_b.unsqueeze(0), sv[b:b+1], self.eps
            ).squeeze(0)
            spectra.append(spec_b)
        return torch.stack(spectra, dim=0)            # (B,F,A)

    def forward(self, X: torch.Tensor, sv: torch.Tensor, correlation: torch.Tensor):
        """
        X: (B,C,F,T)
        sv: (B,F,A,M)
        correlation: kept for signature compatibility (currently unused)

        returns:
          - predict_num_sources=False: doa, spectrum
          - predict_num_sources=True : doa, spectrum, ns, M_hat
        """
        # -------- (optional) predict number of sources --------
        ns = None
        M_hat = None
        if self.predict_num_sources:
            ns_feat = self.encoder_class(X)           # (B,32,?,?)
            ns_feat = ns_feat.view(ns_feat.size(0), -1)
            ns_feat = self.act(self.source_prediction1(ns_feat))
            ns = self.source_prediction2(ns_feat)     # (B, max_sources)
            M_hat = torch.argmax(ns, dim=1)           # (B,)

        # -------- estimate Rx from encoder --------
        z, _ = self.encoder(X)                        # (B,257,h,w)
        z = z.flatten(2)                              # (B,257,h*w)
        z = self.fc(z)                                # (B,257,input_channel^2/2)

        z = z.view(z.size(0), z.size(1), self.input_channel // 2, self.input_channel)
        Rx_real, Rx_imag = torch.chunk(z, chunks=2, dim=-1)
        Rz = torch.complex(Rx_real, Rx_imag)          # (B,257,M,M) if input_channel == 2*M

        # -------- MUSIC --------
        if self.predict_num_sources:
            spectrum = self._pre_music_varM(Rz, sv, M_hat)   # (B,257,A)
        else:
            spectrum = self._pre_music_fixedM(Rz, sv, self.M_fixed)

        # -------- spectrum head --------
        spectrum = self.act(self.convs2(spectrum))     # (B,257,A)
        if self.attention:
            spectrum = self.channel_attention(spectrum)

        spectrum = self.sigmoid(self.convs1(spectrum)).squeeze(1)  # (B,A)
        doa = soft_argmax_peak_refined(spectrum).unsqueeze(1)      # (B,1)

        if self.predict_num_sources:
            return doa, spectrum, ns, M_hat
        return doa, spectrum