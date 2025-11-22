# src/unet_crack_segmentation/training/trainer.py
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(dataloader, desc="Train"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss = self.loss_fn(preds, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0

        for imgs, masks in tqdm(dataloader, desc="Val"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            preds = self.model(imgs)
            loss = self.loss_fn(preds, masks)

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        return {"loss": avg_loss}
