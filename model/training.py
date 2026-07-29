import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class NeuralMusicTrainer:
    """Training loop for the proposed NeuralMUSIC model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        model_dir,
        device="cuda",
        lr_scheduler=None,
        save_best=True,
        source_count_weight=0.05,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.model_dir = model_dir
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.save_best = save_best
        self.source_count_weight = float(source_count_weight)

        self.spectrum_loss = nn.MSELoss()
        self.source_count_loss = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=os.path.join(model_dir, "logs")) if SummaryWriter else None

        os.makedirs(model_dir, exist_ok=True)

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            print(f"Epoch [{epoch + 1}/{self.epochs}]")
            train_loss = self._run_epoch(epoch, train=True)
            val_loss = self._run_epoch(epoch, train=False)

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "best_model.pt"))
                print(f"Saved best model to {self.model_dir}/best_model.pt")

            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "last_model.pt"))

        if self.writer:
            self.writer.close()

    def _run_epoch(self, epoch, train):
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        desc = "Training" if train else "Validating"
        total_loss = 0.0
        valid_steps = 0

        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, targets, sv, correlation = batch[:4]
            inputs = inputs.to(self.device).float()
            sv = sv.to(self.device)
            correlation = correlation.to(self.device)

            if torch.is_tensor(targets):
                targets = targets.to(self.device).float()
            else:
                targets = [target.to(self.device).float() for target in targets]

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = self.model(inputs, sv, correlation)
                loss = self._loss(outputs, targets)
                if train:
                    try:
                        loss.backward()
                        self.optimizer.step()
                    except RuntimeError as exc:
                        print(f"Skip batch after eig/backward error: {exc}")
                        continue

            total_loss += loss.item()
            valid_steps += 1

        avg_loss = total_loss / max(1, valid_steps)
        if self.writer:
            split = "Train" if train else "Validation"
            self.writer.add_scalar(f"Loss/{split}", avg_loss, epoch)
        return avg_loss

    def _loss(self, outputs, targets):
        spectrum = outputs[1] if isinstance(outputs, (tuple, list)) else outputs

        if len(outputs) == 3:
            target_list = list(targets) if not torch.is_tensor(targets) else [row for row in targets]
            spectrum_gt = make_spectrum_targets(target_list, sigma=5, device=self.device)
            source_counts = torch.tensor(
                [len(target) for target in target_list],
                dtype=torch.long,
                device=self.device,
            )
            source_counts = torch.clamp(source_counts, min=0, max=outputs[2].shape[-1] - 1)
            return (
                self.spectrum_loss(spectrum, spectrum_gt)
                + self.source_count_weight * self.source_count_loss(outputs[2], source_counts)
            )

        if not torch.is_tensor(targets):
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        sigma = 10 if targets.shape[1] == 1 else 5
        spectrum_gt = make_spectrum_targets(targets, sigma=sigma, device=self.device)
        return self.spectrum_loss(spectrum, spectrum_gt)


def make_spectrum_targets(targets, sigma, device):
    """Build circular Gaussian MUSIC-spectrum supervision for one or more DOAs."""
    if torch.is_tensor(targets):
        target_list = [row.to(device).float().reshape(-1) for row in targets]
    else:
        target_list = [target.to(device).float().reshape(-1) for target in targets]

    angles = torch.arange(360, device=device, dtype=torch.float32)
    spectra = []
    for doa in target_list:
        doa = doa % 360
        wrapped = torch.cat([doa, doa + 360, doa - 360])
        gauss = torch.exp(-((angles.unsqueeze(0) - wrapped.unsqueeze(1)) ** 2) / (2 * sigma ** 2))
        spectrum = gauss.sum(dim=0)
        spectrum = spectrum / spectrum.max().clamp(min=1e-6)
        spectra.append(spectrum)
    return torch.stack(spectra, dim=0)


def masked_weighted_loss(predicted, target, mask, base_weight=1.0, masked_weight=200.0, loss_type="mse"):
    """Weighted reconstruction loss that emphasizes masked spectrogram bins."""
    if loss_type == "mse":
        loss = F.mse_loss(predicted, target, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(predicted, target, reduction="none")
    else:
        raise ValueError("loss_type must be 'mse' or 'l1'.")

    weight_map = base_weight + mask * (masked_weight - base_weight)
    return (loss * weight_map).mean()


class NeuralMusicPretrainTrainer:
    """Self-supervised masked spectrogram reconstruction trainer."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        model_dir,
        device="cuda",
        lr_scheduler=None,
        save_best=True,
        masked_weight=200.0,
        loss_type="mse",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.model_dir = model_dir
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.save_best = save_best
        self.masked_weight = float(masked_weight)
        self.loss_type = loss_type
        self.writer = SummaryWriter(log_dir=os.path.join(model_dir, "logs")) if SummaryWriter else None

        os.makedirs(model_dir, exist_ok=True)

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            print(f"Pretrain epoch [{epoch + 1}/{self.epochs}]")
            train_loss = self._run_epoch(epoch, train=True)
            val_loss = self._run_epoch(epoch, train=False)

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            print(f"Train recon loss: {train_loss:.6f} | Val recon loss: {val_loss:.6f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "best_model.pt"))
                print(f"Saved best pretrain model to {self.model_dir}/best_model.pt")

            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "last_model.pt"))

        if self.writer:
            self.writer.close()

    def _run_epoch(self, epoch, train):
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        desc = "Pretraining" if train else "Validating"
        total_loss = 0.0

        for masked_spec, target_spec, mask in tqdm(loader, desc=desc, leave=False):
            masked_spec = masked_spec.to(self.device).float()
            target_spec = target_spec.to(self.device).float()
            mask = mask.to(self.device).float()

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                reconstruction = self.model(masked_spec)
                loss = masked_weighted_loss(
                    reconstruction,
                    target_spec,
                    mask,
                    masked_weight=self.masked_weight,
                    loss_type=self.loss_type,
                )
                if train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if self.writer:
            split = "Train" if train else "Validation"
            self.writer.add_scalar(f"PretrainLoss/{split}", avg_loss, epoch)
        return avg_loss
