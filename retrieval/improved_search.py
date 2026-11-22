"""Better search with the same trained CNN: several views of each image, and a correction for hub images."""

import numpy as np
import torch
import torch.nn.functional as F

from common import caption_file, model_file
from data_processing.augment import resize
from data_processing.caption_data import CaptionSplit

# Chosen on val with the 256-d plain CNN, where they took R@1/5/10 from 9.3/26.2/38.9 to 10.7/29.8/43.1.
SEARCH_SIZE = 144
# Besides the whole image: four corner crops and a centre crop, each with this share of the image's side.
CROP_SHARE = 0.85
# A hub image is close to almost every caption. It is measured by this many closest train captions.
NEIGHBOURS = 220
BATCH_SIZE = 64


def image_views(images):
    end = images.shape[-1]
    side = round(end * CROP_SHARE)
    last, middle = end - side, (end - side) // 2
    yield images
    for top, left in ((0, 0), (0, last), (last, 0), (last, last), (middle, middle)):
        yield images[:, :, top:top + side, left:left + side]


@torch.no_grad()
def embed_for_search(model, data, device):
    # One embedding per image, from all of its views and their mirror images.
    model.eval()
    embeddings = []
    for indices in torch.arange(len(data)).split(BATCH_SIZE):
        total = 0
        for view in image_views(data.load_images(indices).to(device).float()):
            view = resize(view, SEARCH_SIZE)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                total = total + model(view).float() + model(view.flip(-1)).float()
        embeddings.append(F.normalize(total, dim=-1))
    return torch.cat(embeddings)


@torch.no_grad()
def hubness(image_embeddings, train_captions):
    # Each image's mean score with its closest train captions. The searched captions are never used.
    device = image_embeddings.device
    closest = torch.empty((0, len(image_embeddings)), device=device)
    # Small chunks, so the 22,047 train images also fit on a 4 GB GPU next to BGE-M3.
    for captions in train_captions.split(1024):
        scores = torch.cat([closest, captions.to(device) @ image_embeddings.T])
        closest = scores.topk(min(NEIGHBOURS, len(scores)), dim=0).values
    return closest.mean(dim=0)


def search_scores(caption_embeddings, image_embeddings, image_hubness):
    return 2 * (caption_embeddings @ image_embeddings.T) - image_hubness


def search_index(model, split, device):
    # What search needs for a split's images (embeddings and hubness), computed once and saved next to the model.
    index_file = model_file().with_name(f"{split}_search_index.pt")
    settings = [SEARCH_SIZE, CROP_SHARE, NEIGHBOURS]
    newest_input = max(model_file().stat().st_mtime, caption_file("train").stat().st_mtime)
    if index_file.exists() and index_file.stat().st_mtime > newest_input:
        saved = torch.load(index_file, map_location=device)
        if saved["settings"] == settings:
            return saved["images"], saved["hubness"]
    print(f"Indexing the {split} images for search...", flush=True)
    images = embed_for_search(model, CaptionSplit(split), device)
    image_hubness = hubness(images, torch.from_numpy(np.load(caption_file("train"))).float())
    torch.save({"images": images.cpu(), "hubness": image_hubness.cpu(), "settings": settings}, index_file)
    return images, image_hubness
