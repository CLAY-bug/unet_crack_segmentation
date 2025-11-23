"""裂缝分割数据集模块
该模块实现了用于裂缝图像语义分割的数据集类，支持图像和掩码的加载、预处理。
"""

import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CrackSegmentationDataset(Dataset):
    """裂缝分割数据集类
    该类继承自PyTorch的Dataset，用于加载和预处理裂缝图像及其对应的分割掩码。
    支持灰度图像的加载、尺寸调整和归一化处理。
    """
    
    def __init__(self, images_dir: str, masks_dir: str, image_size: Tuple[int, int]):
        """初始化裂缝分割数据集
        Args:
            images_dir: 图像根目录
            masks_dir: 掩码根目录
            image_size: 统一resize的尺寸 (H,W)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        # 获取并排序图像文件列表，确保顺序一致性
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    # 告诉Dataloader一共有多少个样本
    def __len__(self):
        return len(self.image_files)

    # 读原图 -> resize -> 归一化
    def _load_image(self, path: str) -> np.ndarray:
        # 以灰度模式读取图像（假设是灰度图）
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 调整图像尺寸，注意cv2使用(width, height)顺序，因此需要反转
        img = cv2.resize(img, self.image_size[::-1])  # cv2 是 (w, h)
        # 转换为float32类型并归一化到[0, 1]范围
        img = img.astype(np.float32) / 255.0
        return img

    # 读mask -> resize -> 二值化
    def _load_mask(self, path: str) -> np.ndarray:
        # 以灰度模式读取掩码图像 0-255
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 使用最近邻插值调整掩码尺寸，避免产生中间值
        # resize时，每个新像素直接拿最近的原像素的值，不做任何混合
        # 例如[0,1]放大一倍后，最近邻会得到[0,0,1,1]，没有中间值
        mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        # 二值化处理：像素值大于127的设为1，否则为0
        mask = (mask > 127).astype(np.float32)
        return mask

    # 根据索引返回一堆（img，mask）
    def __getitem__(self, idx):
        """获取指定索引的数据样本
        该方法根据索引加载对应的图像和掩码，并转换为PyTorch张量格式。
        Args:
            idx: 样本索引
        Returns:
            tuple: 包含两个元素的元组
                - img: 图像张量，形状为(1, H, W)
                - mask: 掩码张量，形状为(1, H, W)
        """
        # 获取图像和掩码的文件名
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        # 构建完整的文件路径
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # 加载图像和掩码
        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # 转换为PyTorch张量并添加通道维度：[H, W] -> [C, H, W]
        # unsqueeze(0)在最前面加一个通道
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

# 测试用的
# dataset = CrackSegmentationDataset(
#     images_dir="./data/processed/train/images",
#     masks_dir="./data/processed/train/masks",
#     image_size=(512, 512),
# )
#
# img, mask = dataset[0]
# print("img:", img.shape, img.dtype, img.min().item(), img.max().item())
# print("mask:", mask.shape, mask.dtype, mask.min().item(), mask.max().item())
