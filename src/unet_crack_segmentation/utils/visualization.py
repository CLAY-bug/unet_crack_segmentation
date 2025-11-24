import os
from typing import Optional
import torch
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