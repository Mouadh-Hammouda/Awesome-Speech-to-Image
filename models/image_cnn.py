import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 models/image_cnn.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn as nn
import torch.nn.functional as F

from common import EMBEDDING_SIZE


class ImageEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, EMBEDDING_SIZE),
            nn.Sigmoid()
        )

    def forward(self, images):
        # 8-bit images, 0 - 255 -> Normalize to 0.0 - 1.0
        return F.normalize(self.layers(images.float() / 255), dim=-1)


if __name__ == "__main__":
    from torchsummary import summary
    from common import IMAGE_SIZE

    summary(ImageEncoderCNN(), (3, IMAGE_SIZE, IMAGE_SIZE), device="cpu")
