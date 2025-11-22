# src/unet_crack_segmentation/datasets/crack_dataset.py
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CrackSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, image_size: Tuple[int, int]):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size

        self.image_files = sorted(os.listdir(images_dir))
        # 默认 mask 文件名与 image 对应
        self.mask_files = self.image_files

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 假设是灰度图
        img = cv2.resize(img, self.image_size[::-1]) # cv2 是 (w, h)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  # 二值化
        return mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # [H, W] -> [C, H, W]
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
