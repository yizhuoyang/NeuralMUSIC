import os
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DeepMucis_plus.dataset.data_processing import generate_music_gt, generate_music_gt_class
from PIL import Image
gradients = {}
import torch
import torch.nn.functional as F


def targets_to_onehot(targets, num_classes):
    batch_size = len(targets)
    targets_flat = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets])
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    batch_indices = torch.arange(batch_size).repeat_interleave(lengths)
    one_hot = torch.zeros((batch_size, num_classes), dtype=torch.float32)
    one_hot[batch_indices, targets_flat] = 1
    return one_hot

def masked_weighted_loss(predicted, target, mask, base_weight=1.0, masked_weight=200.0, loss_type="mse"):
    if loss_type == "mse":
        loss = F.mse_loss(predicted, target, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(predicted, target, reduction='none')
    else:
        raise ValueError("Unsupported loss type. Use 'mse' or 'l1'.")
    weight_map = base_weight + (mask * (masked_weight - base_weight))
    weighted_loss = loss * weight_map
    return weighted_loss.mean()

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch
        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")
            train_loss = self._train_one_epoch(epoch)
            val_loss= self._evaluate(epoch)
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bestmodel")
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        for inputs, targets,sv in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets,sv = inputs.to(self.device).float(), targets.to(self.device).float(),sv.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs,sv)
            if len(outputs)==2:
                if targets.shape[1]==1:
                    sigma = 10
                else:
                    sigma = 5
                spectrum_gt = generate_music_gt(targets,sigma=sigma)
                loss = self.mse_loss(outputs[1],spectrum_gt)
            else:
                loss = self.criterion(outputs, targets)
            try:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            except RuntimeError as e:
                print("Error when calculating noise space")
                continue

        avg_train_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        return avg_train_loss

    def _evaluate(self,epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets,sv in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets,sv = inputs.to(self.device), targets.to(self.device),sv.to(self.device)
                outputs = self.model(inputs,sv)
                if len(outputs)==2:
                    if targets.shape[1]==1:
                        sigma = 10
                    else:
                        sigma = 5
                    spectrum_gt = generate_music_gt(targets,sigma=sigma)
                    loss = self.mse_loss(outputs[1],spectrum_gt)
                else:
                    loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        return avg_val_loss

class ModelTrainer_multiple:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch
        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")
            train_loss = self._train_one_epoch(epoch)
            val_loss= self._evaluate(epoch)
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bestmodel")
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        total_cls = 0
        total_doa = 0
        for inputs, targets,sv,num_source in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, sv,num_source = inputs.to(self.device).float(),sv.to(self.device),num_source.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs,sv)
            if len(outputs)==3:
                sigma = 5
                spectrum_gt = generate_music_gt_class(targets,sigma=sigma).to(self.device)
                loss_doa = self.mse_loss(outputs[1],spectrum_gt)
                loss_class = self.cross_entropy(outputs[2],num_source)
                loss = 0.05*loss_class+loss_doa
            else:
                loss = self.criterion(outputs, targets)
            try:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_doa  += loss_doa.item()
                total_cls  +=loss_class.item()
            except RuntimeError as e:
                print("Error when calculating noise space")
                continue

        avg_train_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls / len(self.train_loader)
        avg_doa_loss = total_doa / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        self.writer.add_scalar("Loss_cls/Train", avg_cls_loss, epoch)
        self.writer.add_scalar("Loss_doa/Train", avg_doa_loss, epoch)
        return avg_train_loss


    def _evaluate(self,epoch):
        self.model.eval()
        total_loss = 0
        total_cls = 0
        total_doa = 0
        with torch.no_grad():
            for inputs, targets,sv,num_source in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, sv,num_source = inputs.to(self.device), sv.to(self.device),num_source.to(self.device)
                outputs = self.model(inputs,sv)
                if len(outputs)==3:
                    sigma = 5
                    spectrum_gt = generate_music_gt_class(targets,sigma=sigma).to(self.device)
                    loss_doa = self.mse_loss(outputs[1],spectrum_gt)
                    loss_class = self.cross_entropy(outputs[2],num_source)
                    loss = 0.005*loss_class+loss_doa
                else:
                    loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_doa  += loss_doa.item()
                total_cls  +=loss_class.item()

        avg_val_loss = total_loss / len(self.val_loader)
        avg_val_doa = total_doa / len(self.val_loader)
        avg_val_cls = total_cls / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        self.writer.add_scalar("Loss_cls/Validation", avg_val_cls, epoch)
        self.writer.add_scalar("Loss_doa/Validation", avg_val_doa, epoch)
        return avg_val_loss



class ModelTrainer_comparison_cls:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch

        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")

            train_loss = self._train_one_epoch(epoch)
            val_loss= self._evaluate(epoch)
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bsetmodel")
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        total_cls = 0
        total_doa = 0
        for inputs, targets,num_source in tqdm(self.train_loader, desc="Training", leave=False):
            inputs,num_source = inputs.to(self.device).float(),num_source.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if len(outputs)==3:
                sigma = 5
                spectrum_gt = generate_music_gt_class(targets,sigma=sigma).to(self.device)
                loss_doa = self.mse_loss(outputs[1],spectrum_gt)
                loss_class = self.cross_entropy(outputs[2],num_source)
                loss = 0.005*loss_class+loss_doa
            else:
                loss = self.criterion(outputs, targets)

            try:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_doa  += loss_doa.item()
                total_cls  +=loss_class.item()

            except RuntimeError as e:
                print("Error when calculating noise space")
                continue

        avg_train_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls / len(self.train_loader)
        avg_doa_loss = total_doa / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        self.writer.add_scalar("Loss_cls/Train", avg_cls_loss, epoch)
        self.writer.add_scalar("Loss_doa/Train", avg_doa_loss, epoch)
        return avg_train_loss


    def _evaluate(self,epoch):
        self.model.eval()
        total_loss = 0
        total_cls = 0
        total_doa = 0
        with torch.no_grad():
            for inputs, targets,num_source in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs,num_source = inputs.to(self.device).float(),num_source.to(self.device)
                outputs = self.model(inputs)
                if len(outputs)==3:
                    sigma = 5
                    spectrum_gt = generate_music_gt_class(targets,sigma=sigma).to(self.device)
                    loss_doa = self.mse_loss(outputs[1],spectrum_gt)
                    loss_class = self.cross_entropy(outputs[2],num_source)
                    loss = 0.005*loss_class+loss_doa
                else:
                    loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_doa  += loss_doa.item()
                total_cls  +=loss_class.item()

        avg_val_loss = total_loss / len(self.val_loader)
        avg_val_doa = total_doa / len(self.val_loader)
        avg_val_cls = total_cls / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        self.writer.add_scalar("Loss_cls/Validation", avg_val_cls, epoch)
        self.writer.add_scalar("Loss_doa/Validation", avg_val_doa, epoch)
        return avg_val_loss

class ModelTrainer_comparison:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch

        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")

            train_loss = self._train_one_epoch(epoch)
            val_loss= self._evaluate(epoch)
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bsetmodel")
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if len(outputs)==2:
                if targets.shape[1]==1:
                    sigma = 10
                else:
                    sigma = 5
                spectrum_gt = generate_music_gt(targets,sigma=sigma)
                loss = self.mse_loss(outputs[1],spectrum_gt)
            else:
                loss = self.criterion(outputs, targets)

            try:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            except RuntimeError as e:
                print("Error when calculating noise space")
                continue

        avg_train_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        return avg_train_loss


    def _evaluate(self,epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                if len(outputs)==2:
                    if targets.shape[1]==1:
                        sigma = 10
                    else:
                        sigma = 5
                    spectrum_gt = generate_music_gt(targets,sigma=sigma)
                    loss = self.mse_loss(outputs[1],spectrum_gt)
                else:
                    loss = self.criterion(outputs, targets)
                if not torch.isnan(loss):
                    total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        return avg_val_loss

class ModelTrainer_pretrain:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch

        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")

            train_loss = self._train_one_epoch()
            val_loss= self._evaluate()
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bestmodel")

            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets,mask in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets,mask = inputs.to(self.device).float(), targets.to(self.device).float(), mask.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = masked_weighted_loss(outputs, targets,mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets,mask in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets,mask = inputs.to(self.device), targets.to(self.device),mask.to(self.device).float()
                outputs = self.model(inputs)
                loss = masked_weighted_loss(outputs, targets,mask)
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        return val_loss
