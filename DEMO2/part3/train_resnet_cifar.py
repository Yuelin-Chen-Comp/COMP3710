"""
Part 3.2 - DAWNBench Challenge: ResNet-18 on CIFAR-10 (Rangpur / CUDA version)

Same model, data pipeline, and training function developed and correctness-tested locally
in Part3_2_DAWNBench.ipynb (Apple Silicon / mps). The only things that changed to run here:
  - device: "mps" -> "cuda"
  - the full 50,000-image training set instead of a 5,000-image Subset
  - use_amp=True (mixed precision now actually activates, since device == "cuda")
  - larger batch size / num_workers, since a real GPU can be fed much faster

Usage:
    python3 train_resnet_cifar.py                      # quick 1-epoch sanity run (demo requirement 2)
    python3 train_resnet_cifar.py --epochs 20 --amp     # full DAWNBench run
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ----------------------------- ResNet-18 (CIFAR-style stem) -----------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        return self.fc(out)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ----------------------------- Training function -----------------------------

def train_resnet18_cifar(train_loader, test_loader, n_epochs, device, use_amp=False, max_lr=0.1):
    model = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9,
                                  weight_decay=5e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=n_epochs * len(train_loader)
    )
    amp_active = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active)

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_active):
                out = model(xb)
                loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        print(f"epoch {epoch:2d}: loss={total_loss/total:.4f}  train_acc={correct/total:.4f}", flush=True)

    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.2f}s on {device} (mixed precision active: {amp_active})")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}")
    return model, train_time, test_acc


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="default 1 = quick sanity/demo run")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-lr", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (requires CUDA)")
    parser.add_argument("--data-root", type=str, default="/tmp/cifar_data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    print("train/test sizes:", len(full_train), len(full_test))

    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, pin_memory=(device == "cuda"))
    test_loader = DataLoader(full_test, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=(device == "cuda"))

    model, train_time, test_acc = train_resnet18_cifar(
        train_loader, test_loader, n_epochs=args.epochs, device=device,
        use_amp=args.amp, max_lr=args.max_lr
    )

    # --- inference demo (lab requirement 2: must run inference live on Rangpur) ---
    model.eval()
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        preds = model(xb.to(device)).argmax(1).cpu()
    classes = full_train.classes
    n_show = min(10, xb.size(0))
    print("\nInference demo (first", n_show, "test images):")
    for i in range(n_show):
        mark = "correct" if preds[i].item() == yb[i].item() else "WRONG"
        print(f"  pred={classes[preds[i]]:<12} true={classes[yb[i]]:<12} {mark}")


if __name__ == "__main__":
    main()
