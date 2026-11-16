import csv
import time
import numpy as np
from functools import partial
from multiprocessing import Pool
from PIL import Image
from utils import DATA_DIR, FEATURES_DIR, available_splits, image_file, print_progress

IMAGE_SIZE = 224


def read_image_ids(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        return sorted({row["image_id"] for row in csv.DictReader(file)})


def load_image(path, size):
    image = Image.open(path)
    image.draft("RGB", (size, size))
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(image).transpose(2, 0, 1)


def cache_split(split, size, pool):
    image_ids = read_image_ids(split)
    paths = [image_file(image_id) for image_id in image_ids]
    images_file = FEATURES_DIR / f"{split}_images_{size}.npy"
    images = np.lib.format.open_memmap(images_file, mode="w+", dtype=np.uint8, shape=(len(paths), 3, size, size))
    start_time = time.monotonic()
    for index, image in enumerate(pool.imap(partial(load_image, size=size), paths, chunksize=64), start=1):
        images[index - 1] = image
        if index % 500 == 0 or index == len(paths):
            print_progress(f"{split} {size}px", index, len(paths), start_time)
    print()
    images.flush()
    (FEATURES_DIR / f"{split}_image_ids.txt").write_text("\n".join(image_ids) + "\n")
    print(f"{split}: {images.shape} saved to {images_file}")


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        for split in available_splits():
            cache_split(split, IMAGE_SIZE, pool)


if __name__ == "__main__":
    main()
