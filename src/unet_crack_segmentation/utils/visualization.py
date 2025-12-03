import os
from typing import Optional,List
import torch
import math
import matplotlib.pyplot as plt



@torch.no_grad()
def visualize_predictions(
        model:torch.nn.Module,
        dataloader, # 一般会传val_loader
        device:torch.device,
        save_dir:str,
        epoch:int,
        max_batches:int = 1, # 最多可视化多少个batch，避免太多图
):
    model.eval()
    os.makedirs(save_dir,exist_ok=True)

    batches_done = 0
    for batch_idx, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        imgs_np = imgs.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()

        batch_size = imgs_np.shape[0]
        for i in range(batch_size):
            fig, axes = plt.subplots(1,3,figsize=(9,3))

            # 原图
            axes[0].imshow(imgs_np[i, 0], cmap="gray")
            axes[0].set_title("image")
            axes[0].axis("off")

            # GT mask
            axes[1].imshow(masks_np[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("mask")
            axes[1].axis("off")

            # 预测 mask
            # axes[2].imshow(preds_np[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[2].imshow(preds_np[i, 0], cmap="gray")
            axes[2].set_title("pred")
            axes[2].axis("off")

            fname = f"epoch{epoch:03d}_batch{batch_idx:03d}_idx.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path,bbox_inches="tight")
            plt.close(fig)
            # 打印前景和背景的概率
            # print(
            #     "probs min/max:",
            #     probs.min().item(),
            #     probs.max().item(),
            # )
        batches_done += 1
        if batches_done >= max_batches:
            break

@torch.no_grad()
def visualize_feature_maps(
        model:torch.nn.Module,
        dataloader,
        device:torch.device,
        save_dir:str,
        epoch:int,
        # 想看的层名列表，比如 ["in_conv", "down1", "down2", "down3", "down4"]
        layer_names:Optional[List[str]]=None,
        max_batches:int=1,
        # 每层最多画多少个通道
        # [B, C, H ,W]， C=通道数(这一层feature maps数量)
        # 例如[1,64,64,64]，通道数为64，如果64个通道全画出来，就是64张小图 太多了
        max_channels:int=8,
):
    """
    可视化指定层的特征图
        1. 用forward hook 在指定层上挂监听
        2. 泡一小批数据
        3. 对每一层、取第一个样本的前 max_channels 个通道，化成网格图保存
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 默认看U-Net的几大模块
    if layer_names is None:
        layer_names = ["in_conv",
                       "down1","down2","down3","down4",
                       "up1","up2","up3","up4"
                       ]

    # 存储各层输出
    feature_maps = {}

    # 定义 hook
    def make_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook

    # 在指定层注册hook
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # 跑一小批数据，让hook抓到输出
    batches_done = 0
    for batch_idx, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        _ = model(imgs) # 触发forward + hook

        batches_done += 1
        if batches_done >= max_batches:
            break

    # 用完记得卸载 hook
    for h in hooks:
        h.remove()

    if not feature_maps:
        print("[visualize_feature_maps] 没有捕获到任何特征图，检查 layer_names 是否正确。")
        return

        # 对每个层画一张图
    for name, fmap in feature_maps.items():
        # fmap: [B, C, H, W]
        b, c, h, w = fmap.shape
        fmap_np = fmap.numpy()

        num_channels = min(c, max_channels)
        # 画第一个样本的前 num_channels 个通道
        nrow = int(math.ceil(num_channels ** 0.5))
        ncol = int(math.ceil(num_channels / nrow))

        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
        # axes 可能是一维也可能二维，统一展平成 list 便于索引
        axes = axes.flatten() if isinstance(axes, (list, tuple)) else axes.ravel()

        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(fmap_np[0, i], cmap="gray")
            ax.set_title(f"{name}: ch={i}")
            ax.axis("off")

        # 多余的子图关掉
        for j in range(num_channels, len(axes)):
            axes[j].axis("off")

        fname = f"epoch{epoch:03d}_layer_{name}.png"
        out_path = os.path.join(save_dir, fname)
        fig.suptitle(f"Epoch {epoch} - {name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[visualize_feature_maps] Saved: {out_path}")