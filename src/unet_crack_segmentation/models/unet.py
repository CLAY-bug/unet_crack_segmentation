# src/unet_crack_segmentation/models/unet.py
import torch
import torch.nn as nn

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
    # 下采样，先最大池化降维度，再进行DoubleConv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x

class UpBlock(nn.Module):
    # 上采样 -> cat skip -> DoubleConv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.double_conv = DoubleConv(2*out_channels,out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip,x], dim=1) # 在通道数方向进行拼接
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        # encoder部分----------------------------------------------------
        # 1. in_conv: Double(in_channels, 64)
        self.in_conv = DoubleConv(in_channels, out_channels=64)
        # 2. 四个DownBlock: 64 -> 128 -> 256 -> 512
        self.down1 = DownBlock(64,128)
        self.down2 = DownBlock(128,256)
        self.down3 = DownBlock(256,512)
        self.down4 = DownBlock(512,1024)
        # decoder部分----------------------------------------------------
        # 3. 四个UpBlock: 1024 -> 512 -> 256 -> 128 -> 64
        self.up1 = UpBlock(1024,512)
        self.up2 = UpBlock(512,256)
        self.up3 = UpBlock(256,128)
        self.up4 = UpBlock(128,64)
        # 4. out_conv: 1*1 Conv， 把64通道映射到nun_classes
        self.out_conv = nn.Conv2d(64,num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码阶段
        # 这里需要保存x1-x5，因为后面会和解码器进行拼接
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 解码阶段（up+skip）
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        # 输出阶段
        logits = self.out_conv(u4)
        return logits

model = UNet(in_channels=1, num_classes=1)
x = torch.randn(1, 1, 512, 512)
y = model(x)
print(y.shape)

