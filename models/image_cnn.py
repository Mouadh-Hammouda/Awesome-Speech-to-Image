import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def conv_bn(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False), nn.BatchNorm2d(out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_path=0.0):
        super().__init__()
        self.drop_path = drop_path
        self.conv1 = conv_bn(in_channels, out_channels, stride)
        self.conv2 = conv_bn(out_channels, out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.conv2(F.relu(self.conv1(x)))
        if self.training and self.drop_path:
            keep = 1 - self.drop_path
            residual = residual * (torch.rand(len(x), 1, 1, 1, device=x.device) < keep).to(residual.dtype) / keep
        return F.relu(residual + self.shortcut(x))


class ImageCNN(nn.Module):
    def __init__(
        self, output_size, widths=(80, 160, 320, 640), depths=(2, 2, 2, 2), stem_width=32, dropout=0.5, drop_path=0.1
    ):
        super().__init__()
        self.checkpoint_blocks = False
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255, persistent=False)
        self.stem = nn.Sequential(conv_bn(3, stem_width, stride=2), nn.ReLU(), conv_bn(stem_width, stem_width), nn.ReLU())

        blocks, channels, block_count = [], stem_width, sum(depths)
        for width, depth in zip(widths, depths):
            for index in range(depth):
                block_drop_path = drop_path * (len(blocks) + 1) / block_count
                blocks.append(ResidualBlock(channels, width, stride=2 if index == 0 else 1, drop_path=block_drop_path))
                channels = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.BatchNorm1d(2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, output_size),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.conv2[1].weight)  # every block starts as the identity

    def forward(self, images):
        x = (images.float() - self.mean) / self.std
        if self.training and self.checkpoint_blocks:
            # Saves memory: the backward pass reruns each block, replaying the random choices of stochastic depth.
            x = checkpoint(self.stem, x, use_reentrant=False)
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks(self.stem(x))
        return F.normalize(self.head(x.mean(dim=(2, 3))), dim=-1)


if __name__ == "__main__":
    model = ImageCNN(output_size=1024)
    images = torch.randint(0, 256, (8, 3, 224, 224), dtype=torch.uint8)
    print("Output shape:", tuple(model(images).shape))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
