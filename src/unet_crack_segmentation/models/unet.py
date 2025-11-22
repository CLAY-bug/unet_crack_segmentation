# src/unet_crack_segmentation/models/unet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        # TODO: 在这里实现 U-Net 的 encoder/decoder + skip connections
        raise NotImplementedError("UNet architecture not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward 逻辑
        raise NotImplementedError
