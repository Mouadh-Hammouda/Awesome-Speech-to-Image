import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 models/image_cnn.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMBEDDING_SIZE, IMAGE_SIZE

# Dilation of the convolutions in blocks 3 and 4: the spacing of the 3 x 3 filter's inputs. The layers and the parameter count stay the same.
# (1, 1) gives each final feature a window of 46 x 46 pixels of the input, (2, 3) one of 86 x 86.
DILATIONS = (1, 1)


def saved_dilations(checkpoint):
    # The weight shapes don't show the dilation, so checkpoints store it; those from before it existed used (1, 1).
    return tuple(checkpoint.get("dilations", (1, 1)))


class ImageEncoderCNN(nn.Module):
    # A plain CNN: four convolution blocks that each halve the image, average pooling, and one linear layer into the caption space.
    def __init__(self, dilations=DILATIONS):
        super().__init__()
        third, fourth = dilations
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=third, dilation=third), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=fourth, dilation=fourth), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            # No activation after this layer: the centred caption vectors have negative values too.
            nn.Linear(256, EMBEDDING_SIZE),
        )

    def forward(self, images):
        # Images are 0 - 255 (8-bit, or float after resizing and augmentation) -> Normalize to 0.0 - 1.0
        return F.normalize(self.layers(images.float() / 255), dim=-1)


if __name__ == "__main__":
    model = ImageEncoderCNN()
    images = torch.randint(0, 256, (8, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    print("Image CNN")
    print(f"  input       {tuple(images.shape[1:])}")
    print(f"  output      {tuple(model(images).shape[1:])}")
    print(f"  dilations   {DILATIONS}")
    print(f"  parameters  {sum(p.numel() for p in model.parameters()):,}")
