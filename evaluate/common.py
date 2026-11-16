import csv
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dataset.cache_images import load_image
from dataset.extract_caption_features import centre, encode_batch, load_encoder
from models.image_cnn import ImageCNN
from utils import DATA_DIR, FEATURES_DIR, get_device, image_file, print_progress

DEFAULT_MODEL_DIR = Path("results/image_cnn")
MODEL_HELP = "results folder with model.pt"
BATCH_SIZE = 64
CROP_LOAD_SCALE = 256 / 224
CROP_SHARE = 0.85
HUBNESS_NEIGHBOURS = 220

device = get_device()


class Split:
    def __init__(self, split, keep_images=None):
        with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        self.base = split
        self.image_ids = sorted(keep_images if keep_images is not None else {row["image_id"] for row in rows})
        self.name = split if keep_images is None else f"{split}{len(self.image_ids)}"
        index = {image_id: position for position, image_id in enumerate(self.image_ids)}
        kept = [(number, row) for number, row in enumerate(rows) if row["image_id"] in index]
        self.rows = np.array([number for number, _ in kept])
        self.captions = [row["caption"].capitalize() for _, row in kept]
        self.caption_images = torch.tensor([index[row["image_id"]] for _, row in kept])
        self.image_captions = [[] for _ in self.image_ids]
        for caption, image in zip(self.captions, self.caption_images.tolist()):
            self.image_captions[image].append(caption)

    def image_file(self, image):
        return image_file(self.image_ids[image])

    def subset(self, count, seed=0):
        chosen = np.random.default_rng(seed).choice(self.image_ids, count, replace=False)
        return Split(self.base, keep_images=chosen.tolist())


class ImageModel:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        self.dir = Path(model_dir)
        self.file = self.dir / "model.pt"
        self.cache_dir = self.dir / "cache"
        checkpoint = torch.load(self.file, map_location=device)
        self.image_size = checkpoint.get("image_size", 224)
        self.epoch = checkpoint["metrics"]["epoch"]
        self.label = f"{self.file}, epoch {self.epoch}"
        model = ImageCNN(checkpoint["model"]["head.4.weight"].shape[0], **checkpoint.get("model_args", {}))
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(device).to(memory_format=torch.channels_last).eval()

    @torch.no_grad()
    def embed(self, split, crops=False):
        cache = self.cache_dir / f"{split.name}_image_embeddings{'_crops' if crops else ''}.pt"
        if cache.exists() and cache.stat().st_mtime > self.file.stat().st_mtime:
            return torch.load(cache, map_location=device)
        load_size = round(self.image_size * CROP_LOAD_SCALE) if crops else self.image_size
        files = [split.image_file(image) for image in range(len(split.image_ids))]
        embeddings, start_time = [], time.monotonic()
        with Pool() as pool:
            images = pool.imap(partial(load_image, size=load_size), files, chunksize=16)
            for start in range(0, len(files), BATCH_SIZE):
                batch = np.stack([next(images) for _ in range(min(BATCH_SIZE, len(files) - start))])
                batch = torch.from_numpy(batch).to(device).float()
                embedding = sum(self._embed_with_mirror(view) for view in self._views(batch, crops))
                embeddings.append(F.normalize(embedding, dim=-1))
                print_progress(f"{split.name} images", start + len(batch), len(files), start_time)
        print()
        embeddings = torch.cat(embeddings)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache)
        return embeddings

    def _views(self, batch, crops):
        if not crops:
            return [batch]
        resize = partial(F.interpolate, size=self.image_size, mode="bilinear", antialias=True, align_corners=False)
        end = batch.shape[-1]
        side = round(end * CROP_SHARE)
        middle = (end - side) // 2
        corners = [(0, 0), (0, end - side), (end - side, 0), (end - side, end - side), (middle, middle)]
        return [resize(batch)] + [resize(batch[:, :, top:top + side, left:left + side]) for top, left in corners]

    def _embed_with_mirror(self, view):
        view = view.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return self.model(view).float() + self.model(view.flip(-1)).float()


def load_model(model_dir=DEFAULT_MODEL_DIR):
    return ImageModel(model_dir)


class CaptionEncoder:
    def __init__(self):
        mean_file = FEATURES_DIR / "caption_mean.npy"
        if not mean_file.exists():
            raise SystemExit(f"{mean_file} is missing: run `python3 -m dataset.extract_caption_features` first.")
        self.mean = np.load(mean_file)
        print("Loading BGE-M3...", flush=True)
        self.model, self.tokenizer = load_encoder(device)

    def __call__(self, captions):
        embeddings = centre(encode_batch(self.model, self.tokenizer, captions, device), self.mean)
        return torch.from_numpy(embeddings).float().to(device)


def caption_embeddings(split):
    embeddings = np.load(FEATURES_DIR / f"{split.base}_text.npy", mmap_mode="r")[split.rows]
    return torch.from_numpy(embeddings).float().to(device)


@torch.no_grad()
def hubness(images, neighbours=HUBNESS_NEIGHBOURS, chunk_size=2048):
    train_captions = np.load(FEATURES_DIR / "train_text.npy", mmap_mode="r")
    closest = torch.empty((0, len(images)), device=device)
    for start in range(0, len(train_captions), chunk_size):
        captions = torch.from_numpy(np.array(train_captions[start:start + chunk_size])).float().to(device)
        scores = torch.cat([closest, captions @ images.T])
        closest = scores.topk(min(neighbours, len(scores)), dim=0).values
    return closest.mean(dim=0)


def evaluation_dir(image_model):
    folder = image_model.dir / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
