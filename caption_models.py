import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from caption_data import IMAGE_SIZE, PADDING

from torchsummary import summary

RESULTS_DIR = Path("results/image_to_caption")
EMBEDDING_SIZE = 256


def model_file(class_names=None):
    # A model trained on only some classes gets its own folder, e.g. cat_dog/, so it never overwrites another.
    if class_names is None:
        return RESULTS_DIR / "model.pt"
    return RESULTS_DIR / "_".join(sorted(name.lower() for name in class_names)) / "model.pt"


class ImageEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, EMBEDDING_SIZE),
        )

    def forward(self, images):
        # 8-bit images, 0 - 255 -> Normalize to 0.0 - 1.0
        return F.normalize(self.layers(images.float() / 255), dim=-1)


class CaptionTextEncoder(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, EMBEDDING_SIZE, padding_idx=PADDING)
        self.projection = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, token_ids):
        # The padding embedding is zero, so summing and dividing by the word count averages the real words.
        word_counts = (token_ids != PADDING).sum(dim=1, keepdim=True).clamp(min=1)
        mean_embedding = self.word_embeddings(token_ids).sum(dim=1) / word_counts
        return F.normalize(self.projection(mean_embedding), dim=-1)


if __name__ == "__main__":
    summary(ImageEncoderCNN(), (3, IMAGE_SIZE, IMAGE_SIZE), device="cpu")
