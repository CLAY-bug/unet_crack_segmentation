# src/unet_crack_segmentation/models/unet.py
import torch
import torch.nn as nn
from torch.nn import MaxPool2d


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            # BN是在通道数上做归一化，所以当前特征图的通道是多少，参数就是多少
            nn.BatchNorm2d(out_channels),
            # inplace=True表示在原tensor上改，节约内存 False就另创建一个tensor
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x

down1 = DownBlock(in_channels=1, out_channels=64)
x = torch.randn(1,1,128,128)
y = down1(x)
print(y.shape)

down2 = DownBlock(in_channels=64, out_channels=128)
x = torch.randn(1, 64, 64, 64)
y = down2(x)
print(y.shape)

down3 = DownBlock(in_channels=128, out_channels=256)
x = torch.randn(1, 128, 32, 32)
y = down3(x)
print(y.shape)



class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        # TODO: 在这里实现 U-Net 的 encoder/decoder + skip connections
        raise NotImplementedError("UNet architecture not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward 逻辑
        raise NotImplementedError
