import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from utils import DATA_DIR, FEATURES_DIR, available_splits, get_device, print_progress

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 256
MAX_TOKENS = 128


def read_captions(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        return [row["caption"] for row in csv.DictReader(file)]


def load_encoder(device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
    return model, tokenizer


@torch.no_grad()
def encode_batch(model, tokenizer, captions, device):
    captions = [caption.capitalize() for caption in captions]
    tokens = tokenizer(captions, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
    cls_tokens = model(**tokens).last_hidden_state[:, 0]
    return F.normalize(cls_tokens.float(), dim=-1).cpu().numpy()


def encode(model, tokenizer, captions, device, split):
    embeddings = []
    start_time = time.monotonic()
    for start in range(0, len(captions), BATCH_SIZE):
        embeddings.append(encode_batch(model, tokenizer, captions[start:start + BATCH_SIZE], device))
        print_progress(split, start + len(embeddings[-1]), len(captions), start_time)
    print()
    return np.concatenate(embeddings)


def centre(embeddings, mean):
    centred = embeddings - mean
    return centred / np.linalg.norm(centred, axis=1, keepdims=True)


def main():
    device = get_device()
    model, tokenizer = load_encoder(device)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = {split: encode(model, tokenizer, read_captions(split), device, split) for split in available_splits()}

    mean = embeddings["train"].mean(axis=0)
    np.save(FEATURES_DIR / "caption_mean.npy", mean)
    for split in available_splits():
        np.save(FEATURES_DIR / f"{split}_text.npy", centre(embeddings[split], mean).astype(np.float32))
        print(f"{split}: {embeddings[split].shape} saved to {FEATURES_DIR / f'{split}_text.npy'}")


if __name__ == "__main__":
    main()
