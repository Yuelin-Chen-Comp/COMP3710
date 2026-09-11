"""Part 4, Task 3 — DCGAN on the real OASIS brain MRI dataset (Rangpur / CUDA).

Same adversarial training loop, non-saturating trick, and snapshot-based progress visualisation
verified locally on MNIST in Part4_Task3_GAN.ipynb; the generator/discriminator are deepened for
the larger OASIS image size.

Usage:  python train_gan_oasis.py --epochs 60 --img-size 64
Outputs: gan_curves.png, gan_progress.png, gan_final_samples.png
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oasis_dataset import OASISImages

LATENT_DIM = 128


class Generator(nn.Module):
    """latent -> 4x4 -> 8 -> 16 -> 32 -> img_size (64), grayscale, tanh output in [-1,1]."""
    def __init__(self, latent_dim=128, img_size=64, ngf=64):
        super().__init__()
        assert img_size == 64, "this generator is sized for 64x64"
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),      # 4 -> 8
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),      # 8 -> 16
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),          # 16 -> 32
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),                # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, img_size=64, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                    # 64 -> 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),  # 32 -> 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),  # 16 -> 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),  # 8 -> 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid(),                            # 4 -> 1
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def sample_grid(G, noise, img_size, n_row=8):
    G.eval()
    with torch.no_grad():
        imgs = ((G(noise).cpu() + 1) / 2).clamp(0, 1)
    G.train()
    n = imgs.size(0); n_col = n // n_row
    canvas = np.zeros((img_size * n_row, img_size * n_col))
    for i in range(n):
        r, c = i // n_col, i % n_col
        canvas[r*img_size:(r+1)*img_size, c*img_size:(c+1)*img_size] = imgs[i, 0].numpy()
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    ds = OASISImages("train", img_size=args.img_size, to_minus_one_one=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True)
    print(f"train slices: {len(ds)}", flush=True)

    G = Generator(LATENT_DIM, args.img_size).to(device)
    D = Discriminator(args.img_size).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}  D params: {sum(p.numel() for p in D.parameters()):,}", flush=True)

    fixed = torch.randn(64, LATENT_DIM, device=device)
    snap_epochs = sorted(set([0, 2, 5, 10, 20, max(0, args.epochs - 1)]))
    snaps = {}
    hist = {"D": [], "G": []}

    t0 = time.time()
    for epoch in range(args.epochs):
        ed, eg, nb = 0.0, 0.0, 0
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            bs = real.size(0)
            ones = torch.ones(bs, 1, device=device)
            zeros = torch.zeros(bs, 1, device=device)

            opt_D.zero_grad()
            loss_d_real = bce(D(real), ones)
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = G(z)
            loss_d_fake = bce(D(fake.detach()), zeros)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            opt_G.zero_grad()
            loss_g = bce(D(fake), ones)   # non-saturating
            loss_g.backward()
            opt_G.step()

            ed += loss_d.item(); eg += loss_g.item(); nb += 1
        hist["D"].append(ed / nb); hist["G"].append(eg / nb)
        print(f"epoch {epoch:2d}: loss_D={ed/nb:.4f} loss_G={eg/nb:.4f}", flush=True)
        if epoch in snap_epochs:
            snaps[epoch] = sample_grid(G, fixed, args.img_size)
    print(f"\nTraining time: {time.time()-t0:.1f}s on {device}", flush=True)

    plt.figure(figsize=(7, 4))
    plt.plot(hist["D"], label="Discriminator"); plt.plot(hist["G"], label="Generator")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("GAN training losses (OASIS)"); plt.tight_layout(); plt.savefig("gan_curves.png", dpi=120)

    ks = sorted(snaps.keys())
    fig, ax = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4.5))
    if len(ks) == 1:
        ax = [ax]
    for a, k in zip(ax, ks):
        a.imshow(snaps[k], cmap="gray"); a.set_title(f"epoch {k}"); a.axis("off")
    plt.suptitle("Generated OASIS brains over training (fixed latent codes)")
    plt.tight_layout(); plt.savefig("gan_progress.png", dpi=120)

    final = sample_grid(G, torch.randn(64, LATENT_DIM, device=device), args.img_size)
    plt.figure(figsize=(8, 8))
    plt.imshow(final, cmap="gray"); plt.xticks(()); plt.yticks(())
    plt.title("Final generator: 64 fresh samples")
    plt.tight_layout(); plt.savefig("gan_final_samples.png", dpi=120)

    print("saved: gan_curves.png gan_progress.png gan_final_samples.png", flush=True)


if __name__ == "__main__":
    main()
