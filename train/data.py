import csv
import numpy as np
import torch
from utils import DATA_DIR, FEATURES_DIR


class CachedSplit:
    def __init__(self, split, image_size):
        image_ids = (FEATURES_DIR / f"{split}_image_ids.txt").read_text().split()
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
            self.caption_images = torch.tensor([image_index[row["image_id"]] for row in csv.DictReader(file)])
        self.images = np.load(FEATURES_DIR / f"{split}_images_{image_size}.npy", mmap_mode="r")
        self.captions = torch.from_numpy(np.load(FEATURES_DIR / f"{split}_text.npy")).float()

    def __len__(self):
        return len(self.images)

    def load_images(self, indices):
        # Reading the memory-mapped file in index order is faster; the result keeps the order of `indices`.
        order = np.argsort(indices.numpy())
        images = np.empty((len(indices), *self.images.shape[1:]), dtype=np.uint8)
        images[order] = self.images[indices.numpy()[order]]
        return torch.from_numpy(images)

    def batch_captions(self, images):
        position = torch.full((len(self),), -1)
        position[images] = torch.arange(len(images))
        captions = torch.nonzero(position[self.caption_images] >= 0).squeeze(1)
        return self.captions[captions], position[self.caption_images[captions]]
