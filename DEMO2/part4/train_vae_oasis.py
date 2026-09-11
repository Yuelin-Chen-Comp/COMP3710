"""Part 4, Task 1 — VAE on the real OASIS brain MRI dataset (Rangpur / CUDA).

Same VAE, loss, and manifold-visualisation approach verified locally on MNIST in
Part4_Task1_VAE.ipynb; the only change is the dataset (OASIS instead of MNIST) and img_size.

Usage:  python train_vae_oasis.py --epochs 30 --img-size 128 --latent-dim 2
Outputs: vae_curves.png, vae_recon.png, vae_latent_scatter.png, vae_manifold.png
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

from oasis_dataset import OASISImages


class VAE(nn.Module):
    def __init__(self, latent_dim=2, in_channels=1, img_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self._feat = img_size // 8
        flat = 128 * self._feat * self._feat
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flat)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z)).view(-1, 128, self._feat, self._feat)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_ds = OASISImages("train", img_size=args.img_size)
    test_ds = OASISImages("test", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"train/test slices: {len(train_ds)} / {len(test_ds)}", flush=True)

    model = VAE(args.latent_dim, 1, args.img_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    hist = {"loss": [], "recon": [], "kl": []}
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, rec, kl, n = 0.0, 0.0, 0.0, 0
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, rl, kll = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()
            tot += loss.item(); rec += rl.item(); kl += kll.item(); n += xb.size(0)
        hist["loss"].append(tot / n); hist["recon"].append(rec / n); hist["kl"].append(kl / n)
        print(f"epoch {epoch:2d}: loss={tot/n:.2f} recon={rec/n:.2f} kl={kl/n:.2f}", flush=True)
    print(f"\nTraining time: {time.time()-t0:.1f}s on {device}", flush=True)

    # ---- curves ----
    plt.figure(figsize=(6, 4))
    for k in ("loss", "recon", "kl"):
        plt.plot(hist[k], label=k)
    plt.xlabel("epoch"); plt.ylabel("loss / image"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("VAE training (OASIS)"); plt.tight_layout(); plt.savefig("vae_curves.png", dpi=120)

    # ---- reconstructions ----
    model.eval()
    xb, _ = next(iter(test_loader))
    xb = xb.to(device)
    with torch.no_grad():
        recon, mu, logvar = model(xb)
    n_show = 8
    fig, ax = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for i in range(n_show):
        ax[0, i].imshow(xb[i, 0].cpu(), cmap="gray"); ax[0, i].axis("off")
        ax[1, i].imshow(recon[i, 0].cpu(), cmap="gray"); ax[1, i].axis("off")
    ax[0, 0].set_title("original", loc="left"); ax[1, 0].set_title("reconstruction", loc="left")
    plt.suptitle("VAE reconstruction (OASIS test slices)"); plt.tight_layout()
    plt.savefig("vae_recon.png", dpi=120)

    # ---- latent scatter (no labels for OASIS -> just density of the encoded test set) ----
    all_mu = []
    with torch.no_grad():
        for xb, _ in test_loader:
            mu, _ = model.encode(xb.to(device))
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu).numpy()
    if args.latent_dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(all_mu[:, 0], all_mu[:, 1], s=4, alpha=0.3)
        plt.xlabel("latent 1"); plt.ylabel("latent 2")
        plt.title("OASIS test set encoded into the 2D latent space")
        plt.tight_layout(); plt.savefig("vae_latent_scatter.png", dpi=120)

        # ---- manifold: decode a grid of latent points ----
        g = 20
        rng = 3.0
        gx = np.linspace(-rng, rng, g)
        gy = np.linspace(-rng, rng, g)
        S = args.img_size
        canvas = np.zeros((S * g, S * g))
        with torch.no_grad():
            for i, yi in enumerate(gy):
                for j, xi in enumerate(gx):
                    z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                    canvas[i*S:(i+1)*S, j*S:(j+1)*S] = model.decode(z).cpu().numpy().reshape(S, S)
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, cmap="gray"); plt.xticks(()); plt.yticks(())
        plt.title("VAE manifold: decoded grid of latent points (OASIS)")
        plt.tight_layout(); plt.savefig("vae_manifold.png", dpi=120)

    print("saved: vae_curves.png vae_recon.png vae_latent_scatter.png vae_manifold.png", flush=True)


if __name__ == "__main__":
    main()
