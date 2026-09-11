"""Part 4, Task 2 — UNet segmentation of the real OASIS brain MRI dataset (Rangpur / CUDA).

Same UNet, categorical (one-hot) output, and per-class Dice metric verified locally on synthetic
shapes in Part4_Task2_UNet.ipynb; the only change is the dataset (OASIS, 4 tissue classes).

Usage:  python train_unet_oasis.py --epochs 25 --img-size 128
Outputs: unet_curves.png, unet_predictions.png, and per-class Dice printed to the log.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from oasis_dataset import OASISSegmentation, N_SEG_CLASSES

CLASS_NAMES = ["background", "CSF", "grey matter", "white matter"]


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes, base=32):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out_conv = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)


def dice_per_class(pred_logits, target, n_classes, eps=1e-6):
    pred = pred_logits.argmax(1)
    out = []
    for c in range(n_classes):
        pc = (pred == c).float()
        tc = (target == c).float()
        inter = (pc * tc).sum()
        out.append(((2 * inter + eps) / (pc.sum() + tc.sum() + eps)).item())
    return out


def evaluate(model, loader, device, n_classes):
    model.eval()
    dices = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            dices.append(dice_per_class(model(xb), yb, n_classes))
    return np.array(dices).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--base", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_ds = OASISSegmentation("train", img_size=args.img_size)
    val_ds = OASISSegmentation("validate", img_size=args.img_size)
    test_ds = OASISSegmentation("test", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"train/val/test slices: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}", flush=True)

    model = UNet(N_SEG_CLASSES, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()   # categorical / one-hot style: per-pixel logits over N classes
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    hist_loss, hist_val_dice = [], []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        hist_loss.append(tot / n)
        vd = evaluate(model, val_loader, device, N_SEG_CLASSES)
        hist_val_dice.append(vd)
        print(f"epoch {epoch:2d}: loss={tot/n:.4f}  val DSC " +
              " ".join(f"{name}={d:.3f}" for name, d in zip(CLASS_NAMES, vd)), flush=True)
    print(f"\nTraining time: {time.time()-t0:.1f}s on {device}", flush=True)

    # ---- final test-set Dice ----
    td = evaluate(model, test_loader, device, N_SEG_CLASSES)
    print("\nFinal TEST-set mean DSC per class:", flush=True)
    for name, d in zip(CLASS_NAMES, td):
        print(f"  {name:<14} {d:.4f}  {'PASS (>0.9)' if d > 0.9 else 'below target'}", flush=True)

    # ---- curves ----
    hv = np.array(hist_val_dice)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist_loss); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].grid(alpha=0.3)
    ax[0].set_title("Training loss")
    for c, name in enumerate(CLASS_NAMES):
        ax[1].plot(hv[:, c], label=name)
    ax[1].axhline(0.9, color="r", ls="--", alpha=0.5)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation DSC"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_title("Per-class validation Dice")
    plt.tight_layout(); plt.savefig("unet_curves.png", dpi=120)

    # ---- prediction visualisation ----
    cmap = ListedColormap(["black", "tomato", "gold", "royalblue"])
    model.eval()
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        preds = model(xb.to(device)).argmax(1).cpu()
    n_show = 4
    fig, ax = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    for i in range(n_show):
        ax[0, i].imshow(xb[i, 0], cmap="gray"); ax[0, i].axis("off"); ax[0, i].set_title("input")
        ax[1, i].imshow(yb[i], cmap=cmap, vmin=0, vmax=3); ax[1, i].axis("off"); ax[1, i].set_title("ground truth")
        ax[2, i].imshow(preds[i], cmap=cmap, vmin=0, vmax=3); ax[2, i].axis("off"); ax[2, i].set_title("UNet prediction")
    plt.suptitle("OASIS segmentation: input / ground truth / prediction"); plt.tight_layout()
    plt.savefig("unet_predictions.png", dpi=120)

    print("saved: unet_curves.png unet_predictions.png", flush=True)


if __name__ == "__main__":
    main()
