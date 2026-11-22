"""Resizes every image to CACHE_SIZE once and stores each split as one uint8 array, so training does not decode JPEGs."""
import csv
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

# Allow direct execution from the repository root: python3 data_processing/cache_images.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from common import CACHE_SIZE, DATA_DIR, FEATURES_DIR, RAW_DIR, SPLITS, print_progress


def read_image_ids(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        return sorted({row["image_id"] for row in csv.DictReader(file)})


def load_image(path, size):
    image = Image.open(path)
    image.draft("RGB", (size, size))
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(image).transpose(2, 0, 1)


def cache_split(split, pool):
    image_ids = read_image_ids(split)
    paths = [RAW_DIR / split / f"{image_id}.jpg" for image_id in image_ids]
    images_file = FEATURES_DIR / f"{split}_images_{CACHE_SIZE}.npy"
    images = np.lib.format.open_memmap(images_file, mode="w+", dtype=np.uint8, shape=(len(paths), 3, CACHE_SIZE, CACHE_SIZE))
    start_time = time.monotonic()
    for index, image in enumerate(pool.imap(partial(load_image, size=CACHE_SIZE), paths, chunksize=64), start=1):
        images[index - 1] = image
        if index % 500 == 0 or index == len(paths):
            print_progress(f"{split} {CACHE_SIZE}px", index, len(paths), start_time)
    print()
    images.flush()
    (FEATURES_DIR / f"{split}_image_ids.txt").write_text("\n".join(image_ids) + "\n")
    print(f"{split}: {images.shape} saved to {images_file}")


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        for split in SPLITS:
            cache_split(split, pool)


if __name__ == "__main__":
    main()
