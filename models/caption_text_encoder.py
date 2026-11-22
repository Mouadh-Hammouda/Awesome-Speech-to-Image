import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from common import EMBEDDING_SIZE, FEATURES_DIR, TEXT_MODEL

MAX_TOKENS = 128
# The mean of the train caption vectors, subtracted from every caption vector (see extract_caption_features.py).
MEAN_FILE = FEATURES_DIR / "caption_mean.npy"
# The fixed 1024 x EMBEDDING_SIZE matrix applied after centring (see reduce_caption_features.py).
PCA_FILE = FEATURES_DIR / f"caption_pca_{EMBEDDING_SIZE}.npy"


def centre(embeddings, mean):
    # Removes the direction all captions share, so the vectors differ by meaning; unit length again afterwards.
    return F.normalize(embeddings - mean, dim=-1)


def project(embeddings, directions):
    # Into the latent space shared with the images; unit length again afterwards.
    return F.normalize(embeddings @ directions, dim=-1)


class CaptionTextEncoder:
    # The pretrained BGE-M3, never trained here: it turns a caption into a vector, and that vector is what the image CNN learns to match.
    def __init__(self, device):
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_MODEL, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def encode(self, captions):
        # Unit-length vectors straight from the model. The stored captions are in capitals, which the model reads worse than sentences.
        tokens = self.tokenizer([caption.capitalize() for caption in captions], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
        return F.normalize(self.model(**tokens).last_hidden_state[:, 0].float(), dim=-1).cpu()

    def embed(self, captions):
        # The vectors used for training and search: encoded, centred with the saved train mean, then reduced with the saved PCA.
        if not MEAN_FILE.exists():
            raise SystemExit(f"{MEAN_FILE} is missing: run python3 data_processing/extract_caption_features.py first.")
        if not PCA_FILE.exists():
            raise SystemExit(f"{PCA_FILE} is missing: run python3 data_processing/reduce_caption_features.py first.")
        centred = centre(self.encode(captions), torch.from_numpy(np.load(MEAN_FILE)))
        return project(centred, torch.from_numpy(np.load(PCA_FILE)))
