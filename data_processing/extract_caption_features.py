"""Encodes every caption once with the pretrained BGE-M3 and stores the centred vectors, row for row like the metadata."""
import csv
import sys
import time
from pathlib import Path

# Allow direct execution from the repository root: python3 data_processing/extract_caption_features.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from common import DATA_DIR, FEATURES_DIR, SPLITS, get_device, print_progress
from models.caption_text_encoder import MEAN_FILE, CaptionTextEncoder, centre

BATCH_SIZE = 256


def read_captions(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        return [row["caption"] for row in csv.DictReader(file)]


def encode(text_encoder, captions, split):
    embeddings = []
    start_time = time.monotonic()
    for start in range(0, len(captions), BATCH_SIZE):
        embeddings.append(text_encoder.encode(captions[start:start + BATCH_SIZE]))
        print_progress(split, start + len(embeddings[-1]), len(captions), start_time)
    print()
    return torch.cat(embeddings)


def main():
    text_encoder = CaptionTextEncoder(get_device())
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = {split: encode(text_encoder, read_captions(split), split) for split in SPLITS}

    # Centred with the train mean, which search and any later stage must reuse: it is saved next to the vectors.
    mean = embeddings["train"].mean(dim=0)
    np.save(MEAN_FILE, mean.numpy())
    for split in SPLITS:
        np.save(FEATURES_DIR / f"{split}_text.npy", centre(embeddings[split], mean).numpy().astype(np.float32))
        print(f"{split}: {tuple(embeddings[split].shape)} saved to {FEATURES_DIR / f'{split}_text.npy'}")


if __name__ == "__main__":
    main()
