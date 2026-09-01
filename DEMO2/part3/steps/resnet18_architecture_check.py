import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """The basic residual block used in ResNet-18/34 (two 3x3 convs + a skip connection)."""
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
        # CIFAR-style stem: 3x3 conv, stride 1, NO max-pool (unlike the ImageNet ResNet-18
        # stem, which uses a 7x7 stride-2 conv + maxpool -- that would shrink our already-tiny
        # 32x32 CIFAR images too aggressively before any residual blocks even run).
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


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    x = torch.randn(8, 3, 32, 32)   # batch of 8 fake CIFAR10-sized images
    out = model(x)
    print("Output shape:", out.shape)   # should be (8, 10)
    assert out.shape == (8, 10)

    # verify backward pass works too
    loss = out.sum()
    loss.backward()
    print("Backward pass OK, all gradients populated:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))
