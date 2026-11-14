import torch.nn.functional as F
from torch import nn

from config import TEXT_EMBEDDING_SIZE
from utils import get_image_transform

image_transform = get_image_transform()


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512)
        )

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, TEXT_EMBEDDING_SIZE)
        )

    def forward(self, images):
        features = self.features(images)
        embeddings = self.projection(features)

        return F.normalize(embeddings, p=2, dim=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
