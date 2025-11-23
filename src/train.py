"""训练入口脚本

整体流程：Yaml配置 -> Dataset/Dataloader -> UNet -> Loss -> Trainer

该文件负责完整的训练流程组织，包括：
1) 读取配置文件并解析超参数
2) 构建训练/验证数据集与 DataLoader
3) 实例化 UNet 模型、优化器与损失函数
4) 通过 Trainer 执行训练与验证，并打印损失指标

配置文件默认路径为 `./configs/default.yaml`，常用字段：
- `data.root`：数据根目录
- `data.train_images_dir` / `data.train_masks_dir`：训练集图像/掩码相对路径
- `data.val_images_dir` / `data.val_masks_dir`：验证集图像/掩码相对路径
- `data.image_size`：输入统一缩放尺寸，形如 `[H, W]`
- `data.batch_size`：每批样本数
- `data.num_workers`：DataLoader 工作线程数（Windows 需在主入口保护下使用）
- `model.in_channels`：输入通道数（灰度图为 1，RGB 为 3）
- `model.num_classes`：类别数（二分类分割通常为 1）
- `train.lr` / `train.weight_decay`：优化器超参数
- `train.epochs`：训练总轮数
"""

# -----------------------------
# 基础库与项目组件导入
# -----------------------------
import os  # 路径拼接与文件系统操作
import torch
from torch.utils.data import DataLoader  # 批量数据加载器
from torch import nn  # 常用损失函数与网络组件

from unet_crack_segmentation.config import load_config  # 配置加载函数
from unet_crack_segmentation.datasets.crack_dataset import CrackSegmentationDataset  # 裂缝分割数据集
from unet_crack_segmentation.models.unet import UNet  # UNet 模型定义
from unet_crack_segmentation.training.trainer import Trainer  # 训练/验证控制器


def main():
    """训练流程主函数

    步骤概览：
    - 加载 YAML 配置，统一管理数据与训练超参数
    - 自动选择计算设备（优先使用 CUDA）
    - 构建训练/验证数据集与 DataLoader
    - 初始化 UNet 模型、优化器与损失函数
    - 逐轮训练与验证，输出损失指标
    """

    # 读取配置文件（默认位于项目根目录下的 `configs/default.yaml`）
    cfg = load_config("./configs/default.yaml")
    # 自动选择设备：如可用则使用 GPU，否则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建训练数据集：提供图像/掩码路径与统一的输入尺寸
    train_dataset = CrackSegmentationDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.train_images_dir),  # 训练图像目录
        masks_dir=os.path.join(cfg.data.root, cfg.data.train_masks_dir),    # 训练掩码目录
        image_size=tuple(cfg.data.image_size),                              # 统一缩放尺寸 (H, W)
    )
    # 构建验证数据集：配置与训练集一致，仅路径不同
    val_dataset = CrackSegmentationDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.val_images_dir),    # 验证图像目录
        masks_dir=os.path.join(cfg.data.root, cfg.data.val_masks_dir),      # 验证掩码目录
        image_size=tuple(cfg.data.image_size),                              # 统一缩放尺寸 (H, W)
    )

    # DataLoader 负责批量化与并行加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,  # 每批样本数
        shuffle=True,                    # 训练集打乱以提升泛化
        num_workers=cfg.data.num_workers # 数据加载并行度（Windows 需主入口保护）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,  # 与训练集保持一致
        shuffle=False,                   # 验证集不打乱，保证可复现评估
        num_workers=cfg.data.num_workers
    )

    # 初始化 UNet 模型：输入通道数与类别数由配置决定
    model = UNet(
        in_channels=cfg.model.in_channels, # 灰度图为1， RGB图为3
        num_classes=cfg.model.num_classes,
    )

    # 优化器选用 Adam：学习率与权重衰减来自配置
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    # 先占位：后续可替换为 BCE + Dice 组合损失以提升稳定性与收敛
    loss_fn = nn.BCEWithLogitsLoss()  # 二值交叉熵（带 Logits），常用于二分类分割

    # 封装训练/验证流程的控制器，负责前向/反向与评估
    trainer = Trainer(model, optimizer, loss_fn, device)

    # 训练主循环：从第 1 轮到 `epochs`，每轮先训练后验证
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)  # 单轮训练，返回指标字典（如 loss）
        val_metrics = trainer.validate(val_loader)             # 验证评估，返回指标字典

        # 打印当前轮的训练与验证损失，便于监控收敛情况
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}"
        )


# Windows 下使用多进程 DataLoader 需要主入口保护
if __name__ == "__main__":
    main()
