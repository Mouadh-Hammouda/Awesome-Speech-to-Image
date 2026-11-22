"""The cached images and caption vectors of a split."""
import csv
import numpy as np
import torch
from common import CACHE_SIZE, CLASSES, DATA_DIR, FEATURES_DIR, caption_file


# An image has a class only if at least this many of its captions name it: a class word in a single caption is often a background detail.
MIN_CAPTION_VOTES = 2


def class_votes(captions):
    # For each class in CLASSES, how many of the captions (each a set of words) name it.
    return [sum(bool(words & class_words) for words in captions) for class_words in CLASSES.values()]


def class_labels(captions):
    # Every class with enough votes: an image can have several classes (a dog in a car) or none.
    return [votes >= MIN_CAPTION_VOTES for votes in class_votes(captions)]


def main_class(captions):
    # The index of the one class most captions name, or None if no class has enough votes or two classes tie; for plots that need one colour per image.
    votes = class_votes(captions)
    best = max(range(len(votes)), key=votes.__getitem__)
    clear = votes[best] >= MIN_CAPTION_VOTES and sorted(votes)[-2] < votes[best]
    return best if clear else None


class CaptionSplit:
    # Images are the uint8 arrays from cache_images.py, read from disk as needed; captions are the reduced BGE-M3 vectors from reduce_caption_features.py.
    def __init__(self, split):
        for file in (FEATURES_DIR / f"{split}_image_ids.txt", FEATURES_DIR / f"{split}_images_{CACHE_SIZE}.npy", caption_file(split)):
            if not file.exists():
                raise SystemExit(f"{file} is missing: run the data steps in README.md first (cache_images, extract_caption_features, reduce_caption_features).")
        image_ids = (FEATURES_DIR / f"{split}_image_ids.txt").read_text().split()
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        # Row i of the captions is row i of the metadata; caption_images[i] is the image it describes.
        self.caption_images = torch.tensor([image_index[row["image_id"]] for row in rows])
        self.captions = torch.from_numpy(np.load(caption_file(split))).float()
        self.images = np.load(FEATURES_DIR / f"{split}_images_{CACHE_SIZE}.npy", mmap_mode="r")
        # Each image's classes (one column per class in CLASSES) from the words of its captions, for the class-level metrics.
        image_captions = [[] for _ in image_ids]
        for row, image in zip(rows, self.caption_images.tolist()):
            image_captions[image].append(set(row["caption"].split()))
        self.image_classes = torch.tensor([class_labels(captions) for captions in image_captions])
        # One class per image for the confusion matrix: main_class, or len(CLASSES) (OTHER) when there is none.
        main_classes = [main_class(captions) for captions in image_captions]
        self.image_main_classes = torch.tensor([len(CLASSES) if index is None else index for index in main_classes])

    def __len__(self):
        return len(self.images)

    def load_images(self, indices):
        # Reading the memory-mapped file in index order is faster; the result keeps the order of `indices`.
        order = np.argsort(indices.numpy())
        images = np.empty((len(indices), *self.images.shape[1:]), dtype=np.uint8)
        images[order] = self.images[indices.numpy()[order]]
        return torch.from_numpy(images)

    def batch_captions(self, images):
        # All the captions of the given images, and for each one the position in `images` of the image it describes.
        position = torch.full((len(self),), -1)
        position[images] = torch.arange(len(images))
        captions = torch.nonzero(position[self.caption_images] >= 0).squeeze(1)
        return self.captions[captions], position[self.caption_images[captions]]
