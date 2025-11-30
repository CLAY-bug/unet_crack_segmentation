"""
训练器模块
该模块实现了用于模型训练和验证的Trainer类，封装了训练循环、验证循环和损失计算逻辑。
"""

from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


class Trainer:
    """
    模型训练器类
    该类封装了深度学习模型的训练和验证流程，支持自定义模型、优化器、损失函数和设备。
    提供了单个epoch的训练和验证方法，自动处理前向传播、反向传播和参数更新。
    Attributes:
        model: 神经网络模型
        optimizer: 优化器
        loss_fn: 损失函数
        device: 训练设备（CPU或GPU）
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ):
        # 将模型移动到指定设备（CPU或GPU）
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个完整的epoch包括前向传播、损失计算、反向传播和参数更新。
        使用tqdm显示训练进度条。
        Args: dataloader: 训练数据加载器
        Returns:
            Dict[str, float]: 包含训练指标的字典，格式为 {"loss": 平均损失值}
        """
        # 设置模型为训练模式（启用dropout、batch normalization等）
        self.model.train()
        total_loss = 0.0

        # 遍历数据批次，使用tqdm显示进度，desc用于描述进度条文字
        for imgs, masks in tqdm(dataloader, desc="Train"):
            # 将图像和掩码移动到指定设备
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            # 清空梯度
            self.optimizer.zero_grad()
            # 前向传播：模型预测
            preds = self.model(imgs)
            # 计算损失
            loss = self.loss_fn(preds, masks)
            # 反向传播：计算梯度
            loss.backward()
            # 更新模型参数
            self.optimizer.step()

            # 累积总损失（乘以batch大小以便后续计算平均值）
            total_loss += loss.item() * imgs.size(0)

        # 计算整个epoch的平均损失
        avg_loss = total_loss / len(dataloader.dataset)
        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证模型性能
        该方法在验证集上评估模型性能，不进行梯度计算和参数更新。
        使用@torch.no_grad()装饰器禁用梯度计算以节省内存和加速推理。
        Args:dataloader: 验证数据加载器
        Returns:
            Dict[str, float]: 包含验证指标的字典，格式为 {"loss": 平均损失值}
        """
        # 设置模型为评估模式（禁用dropout、batch normalization等）
        self.model.eval()
        total_loss = 0.0

        # 遍历验证数据批次，使用tqdm显示进度
        for imgs, masks in tqdm(dataloader, desc="Val"):
            # 将图像和掩码移动到指定设备
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            # 前向传播：模型预测（无梯度计算）
            preds = self.model(imgs)
            # 计算损失
            loss = self.loss_fn(preds, masks)

            # 累积总损失（乘以batch大小以便后续计算平均值）
            total_loss += loss.item() * imgs.size(0)

        # 计算整个验证集的平均损失
        avg_loss = total_loss / len(dataloader.dataset)
        return {"loss": avg_loss}
