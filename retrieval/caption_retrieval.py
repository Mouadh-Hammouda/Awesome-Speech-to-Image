"""Shared model loading and image embeddings for search and diagnostics."""

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE_SIZE, FEATURES_DIR, RAW_DIR, get_device, model_file
from data_processing.augment import resize
from models.image_cnn import ImageEncoderCNN, saved_dilations

BATCH_SIZE = 128

device = get_device()


def load_image_encoder():
    checkpoint = torch.load(model_file(), map_location=device)
    model = ImageEncoderCNN(saved_dilations(checkpoint))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def embed_split(image_encoder, split):
    trained_file = model_file()
    embeddings_file = trained_file.parent / f"{split}_image_embeddings.pt"
    image_ids = (FEATURES_DIR / f"{split}_image_ids.txt").read_text().split()
    # Saved embeddings are out of date once the model has been retrained.
    if embeddings_file.exists() and embeddings_file.stat().st_mtime > trained_file.stat().st_mtime:
        image_embeddings = torch.load(embeddings_file, map_location=device)
    else:
        images = np.load(FEATURES_DIR / f"{split}_images_{CACHE_SIZE}.npy", mmap_mode="r")
        embeddings = []
        for start in range(0, len(images), BATCH_SIZE):
            batch = resize(torch.from_numpy(np.ascontiguousarray(images[start:start + BATCH_SIZE])).to(device))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                # An image and its mirror image give one embedding together, as in training's evaluation.
                embedding = image_encoder(batch).float() + image_encoder(batch.flip(-1)).float()
            embeddings.append(F.normalize(embedding, dim=-1))
        image_embeddings = torch.cat(embeddings)
        torch.save(image_embeddings.cpu(), embeddings_file)
    return [RAW_DIR / split / f"{image_id}.jpg" for image_id in image_ids], image_embeddings
