"""Live inference demo for Part 4 Task 2 (UNet on OASIS) -- lab requirement: "You must run
inference at demonstration on a test set and show the model is working correctly during the demo."

Loads the checkpoint saved by train_unet_oasis.py (unet_oasis.pt) and runs inference on a batch
of real OASIS test slices -- no training, so this finishes in a few seconds. Prints per-class Dice
for this batch and saves a visualisation (input / ground truth / prediction) to inspect live.

Usage: python infer_unet_oasis.py
"""
import time
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from train_unet_oasis import UNet, dice_per_class, CLASS_NAMES
from oasis_dataset import OASISSegmentation, N_SEG_CLASSES


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    ckpt = torch.load("unet_oasis.pt", map_location=device)
    model = UNet(N_SEG_CLASSES, base=ckpt["base"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded unet_oasis.pt (img_size={ckpt['img_size']}, base={ckpt['base']})", flush=True)

    test_ds = OASISSegmentation("test", img_size=ckpt["img_size"])
    loader = DataLoader(test_ds, batch_size=16, shuffle=True, num_workers=4)
    xb, yb = next(iter(loader))

    t0 = time.time()
    with torch.no_grad():
        xb_dev, yb_dev = xb.to(device), yb.to(device)
        logits = model(xb_dev)
        preds = logits.argmax(1)
    infer_time = time.time() - t0
    print(f"Inference on {xb.size(0)} test slices took {infer_time:.3f}s", flush=True)

    dices = dice_per_class(logits, yb_dev, N_SEG_CLASSES)
    print("\nDSC on this live test batch:", flush=True)
    for name, d in zip(CLASS_NAMES, dices):
        print(f"  {name:<14} {d:.4f}  {'PASS (>0.9)' if d > 0.9 else 'below target'}", flush=True)

    cmap = ListedColormap(["black", "tomato", "gold", "royalblue"])
    n_show = 4
    fig, ax = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    for i in range(n_show):
        ax[0, i].imshow(xb[i, 0], cmap="gray"); ax[0, i].axis("off"); ax[0, i].set_title("input MRI slice")
        ax[1, i].imshow(yb[i], cmap=cmap, vmin=0, vmax=3); ax[1, i].axis("off"); ax[1, i].set_title("ground truth")
        ax[2, i].imshow(preds[i].cpu(), cmap=cmap, vmin=0, vmax=3); ax[2, i].axis("off"); ax[2, i].set_title("live prediction")
    plt.suptitle("Live UNet inference on real OASIS test slices")
    plt.tight_layout()
    plt.savefig("unet_live_inference.png", dpi=120)
    print("\nsaved: unet_live_inference.png", flush=True)


if __name__ == "__main__":
    main()
