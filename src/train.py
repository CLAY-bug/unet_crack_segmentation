# src/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn

from unet_crack_segmentation.config import load_config
from unet_crack_segmentation.datasets.crack_dataset import CrackSegmentationDataset
from unet_crack_segmentation.models.unet import UNet
from unet_crack_segmentation.training.trainer import Trainer

def main():
    cfg = load_config("./configs/default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CrackSegmentationDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.train_images_dir),
        masks_dir=os.path.join(cfg.data.root, cfg.data.train_masks_dir),
        image_size=tuple(cfg.data.image_size),
    )
    val_dataset = CrackSegmentationDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.val_images_dir),
        masks_dir=os.path.join(cfg.data.root, cfg.data.val_masks_dir),
        image_size=tuple(cfg.data.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = UNet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # 先占位：后面换成 BCE + Dice 组合
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, optimizer, loss_fn, device)

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}")

if __name__ == "__main__":
    main()
