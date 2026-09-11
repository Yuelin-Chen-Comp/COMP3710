"""Shared OASIS dataset loaders for Part 4 (VAE / UNet / GAN) on Rangpur.

Preprocessed OASIS at /home/groups/comp3710/OASIS/:
  keras_png_slices_{train,validate,test}/       case_XXX_slice_Y.nii.png  (256x256 uint8 grayscale)
  keras_png_slices_seg_{train,validate,test}/   seg_XXX_slice_Y.nii.png   (256x256 uint8, values {0,85,170,255})

The 4 segmentation values 0/85/170/255 map to class indices 0/1/2/3 via `mask // 85`.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

OASIS_ROOT = "/home/groups/comp3710/OASIS"
N_SEG_CLASSES = 4


class OASISImages(Dataset):
    """Images only (for the VAE and the GAN) — no labels needed."""
    def __init__(self, split="train", img_size=128, to_minus_one_one=False):
        self.dir = os.path.join(OASIS_ROOT, f"keras_png_slices_{split}")
        self.files = sorted(f for f in os.listdir(self.dir) if f.endswith(".png"))
        self.img_size = img_size
        self.to_minus_one_one = to_minus_one_one   # True -> [-1,1] for tanh GANs, else [0,1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.files[idx])).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.to_minus_one_one:
            x = x * 2.0 - 1.0
        return x, 0


class OASISSegmentation(Dataset):
    """(image, class-index mask) pairs for the UNet."""
    def __init__(self, split="train", img_size=128):
        self.img_dir = os.path.join(OASIS_ROOT, f"keras_png_slices_{split}")
        self.seg_dir = os.path.join(OASIS_ROOT, f"keras_png_slices_seg_{split}")
        self.img_files = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".png"))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        seg_name = img_name.replace("case_", "seg_")   # case_001_slice_0.nii.png -> seg_001_slice_0.nii.png

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        seg = Image.open(os.path.join(self.seg_dir, seg_name)).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            seg = seg.resize((self.img_size, self.img_size), Image.NEAREST)   # NEAREST: don't blend class labels

        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        m = torch.from_numpy((np.array(seg, dtype=np.int64)) // 85)   # {0,85,170,255} -> {0,1,2,3}
        return x, m
