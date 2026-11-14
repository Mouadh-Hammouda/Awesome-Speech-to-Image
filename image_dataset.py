import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config import DATA_DIR, IMAGE_CACHE_DIR, IMAGE_SIZE, TEXT_EMBEDDING_SIZE
from utils import get_image_path, get_image_transform, load_metadata

image_transform = get_image_transform()


class ImageCaptionDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.rows = load_metadata(split)
        embeddings = np.load(DATA_DIR / f"{split}_bge_m3.npy", allow_pickle=False)
        if embeddings.shape != (len(self.rows), TEXT_EMBEDDING_SIZE):
            raise ValueError("Metadata and embedding shapes disagree")
        self.targets = torch.from_numpy(embeddings).float()

        # Captions sharing an image use the same cached pixels.
        self.image_ids = list(dict.fromkeys(row["image_id"] for row in self.rows))
        self.image_index_by_id = {image_id: i for i, image_id in enumerate(self.image_ids)}
        self.cached_images = self._load_image_cache()

    def _load_image_cache(self):
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = IMAGE_CACHE_DIR / f"{self.split}_images.npy"
        metadata_path = IMAGE_CACHE_DIR / f"{self.split}_images.json"
        shape = (len(self.image_ids), 3, IMAGE_SIZE, IMAGE_SIZE)

        # Rebuild if image order, source files, or preprocessing settings change.
        sources = []
        for image_id in self.image_ids:
            stat = get_image_path(self.split, image_id).stat()
            sources.append([image_id, stat.st_size, stat.st_mtime_ns])
        metadata = {"sources": sources, "transform": repr(image_transform), "shape": list(shape)}

        if (
            cache_path.exists()
            and metadata_path.exists()
            and json.loads(metadata_path.read_text()) == metadata
        ):
            print(f"{self.split}: reusing cached images")
            return np.load(cache_path, mmap_mode="r")

        # Write a temporary file so an interrupted run leaves the old cache intact.
        temporary_path = cache_path.with_suffix(".tmp.npy")
        cached = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float32, shape=shape)
        for index, image_id in enumerate(tqdm(self.image_ids, desc=f"Preprocess {self.split}", unit="image")):
            with Image.open(get_image_path(self.split, image_id)) as image:
                cached[index] = image_transform(image.convert("RGB")).numpy()
        cached.flush()
        del cached
        temporary_path.replace(cache_path)
        metadata_path.write_text(json.dumps(metadata))
        return np.load(cache_path, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        pixels = self.cached_images[self.image_index_by_id[row["image_id"]]]
        image_tensor = torch.from_numpy(pixels.copy())

        return image_tensor, self.targets[index], row["image_id"]
