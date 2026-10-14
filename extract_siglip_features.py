import csv
import numpy as np
import torch
import time
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
SPLITS = ["train", "val"]
MODEL_NAME = "google/siglip2-base-patch16-224"
BATCH_SIZE = 64


def load_siglip():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device, dtype


def print_progress(split, done, total, start_time):
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{split}: {done}/{total}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r")


@torch.no_grad()
def encode_images(siglip, split, image_ids):
    model, processor, device, dtype = siglip
    embeddings = {}
    start_time = time.monotonic()
    for start in range(0, len(image_ids), BATCH_SIZE):
        batch_ids = image_ids[start:start + BATCH_SIZE]
        images = [Image.open(RAW_DIR / split / f"{image_id}.jpg").convert("RGB") for image_id in batch_ids]
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = model.get_image_features(pixel_values=inputs["pixel_values"].to(dtype)).pooler_output
        features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
        embeddings.update(zip(batch_ids, features))
        print_progress(split, len(embeddings), len(image_ids), start_time)
    print()
    return embeddings


def encode_split(siglip, split):
    with open(OUTPUT_DIR / f"{split}_meta.csv", newline="") as file:
        row_image_ids = [row["image_id"] for row in csv.DictReader(file)]
    # Each image has ~5 captions: encode every unique image once, then look it up per row.
    embeddings = encode_images(siglip, split, sorted(set(row_image_ids)))
    return np.stack([embeddings[image_id] for image_id in row_image_ids])


def main():
    siglip = load_siglip()
    for split in SPLITS:
        np.save(OUTPUT_DIR / f"{split}_y.npy", encode_split(siglip, split))


if __name__ == "__main__":
    main()
