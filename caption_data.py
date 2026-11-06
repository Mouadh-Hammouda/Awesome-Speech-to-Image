import csv
import numpy as np
import random
import torch
from collections import Counter
from PIL import Image
from utils import DATA_DIR, RAW_DIR

LOAD_SIZE = 140
IMAGE_SIZE = 128
MIN_WORD_COUNT = 5
MAX_CAPTION_WORDS = 40
PADDING, UNKNOWN = 0, 1


def build_vocabulary(captions):
    # Built from the train captions only; rare words share the unknown token.
    word_counts = Counter(word for caption in captions for word in caption.split())
    words = sorted(word for word, count in word_counts.items() if count >= MIN_WORD_COUNT)
    return {word: index for index, word in enumerate(words, start=2)}


def tokenise(captions, vocabulary):
    token_ids = torch.full((len(captions), MAX_CAPTION_WORDS), PADDING)
    for row, caption in enumerate(captions):
        words = caption.split()[:MAX_CAPTION_WORDS]
        token_ids[row, :len(words)] = torch.tensor([vocabulary.get(word, UNKNOWN) for word in words], dtype=torch.long)
    return token_ids


def load_images(split, image_ids):
    images = [Image.open(RAW_DIR / split / f"{image_id}.jpg").convert("RGB").resize((LOAD_SIZE, LOAD_SIZE)) for image_id in image_ids]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)


class CaptionSplit:
    def __init__(self, split, vocabulary=None):
        with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        captions = [row["caption"] for row in rows]
        self.vocabulary = vocabulary or build_vocabulary(captions)
        self.captions = tokenise(captions, self.vocabulary)
        row_image_ids = [row["image_id"] for row in rows]
        image_ids = sorted(set(row_image_ids))
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        self.caption_images = torch.tensor([image_index[image_id] for image_id in row_image_ids])
        self.image_captions = [[] for _ in image_ids]
        for row, image in enumerate(self.caption_images.tolist()):
            self.image_captions[image].append(row)
        self.images = load_images(split, image_ids)


def crop(images, top, left):
    return images[:, :, top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]


def augment(images):
    images = crop(images, random.randint(0, LOAD_SIZE - IMAGE_SIZE), random.randint(0, LOAD_SIZE - IMAGE_SIZE))
    return torch.where(torch.rand(len(images), 1, 1, 1) < 0.5, images.flip(3), images)


def centre_crop(images):
    margin = (LOAD_SIZE - IMAGE_SIZE) // 2
    return crop(images, margin, margin)
