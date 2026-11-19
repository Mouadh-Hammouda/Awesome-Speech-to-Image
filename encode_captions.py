import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import (
    CAPTION_BASE_EMBEDDING_SIZE, CAPTION_BATCH_SIZE, CAPTION_DEVICE,
    CAPTION_MODEL_NAME, CAPTION_PROJECTION_PATH, PROJECTION_BATCH_SIZE,
    PROJECTION_EPOCHS, PROJECTION_LEARNING_RATE, SEED, SPLITS, TEXT_EMBEDDING_SIZE,
)
from utils import caption_embedding_path, load_metadata


def load_caption_encoder(projected=True):
    from sentence_transformers import SentenceTransformer, models

    encoder = SentenceTransformer(CAPTION_MODEL_NAME, device=CAPTION_DEVICE)
    if projected and TEXT_EMBEDDING_SIZE != CAPTION_BASE_EMBEDDING_SIZE:
        dense = models.Dense(
            CAPTION_BASE_EMBEDDING_SIZE, TEXT_EMBEDDING_SIZE,
            bias=False, activation_function=nn.Identity(),
        )
        dense.linear.load_state_dict(torch.load(CAPTION_PROJECTION_PATH, map_location="cpu", weights_only=True))
        encoder.add_module("projection", dense.to(encoder.device))
        encoder.add_module("normalize", models.Normalize())
    return encoder.requires_grad_(False).eval()


def train_projection(embeddings):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projection = nn.Linear(CAPTION_BASE_EMBEDDING_SIZE, TEXT_EMBEDDING_SIZE, bias=False).to(device)
    nn.init.orthogonal_(projection.weight)
    optimizer = torch.optim.Adam(projection.parameters(), lr=PROJECTION_LEARNING_RATE)
    loader = DataLoader(torch.from_numpy(embeddings).float(), batch_size=PROJECTION_BATCH_SIZE, shuffle=True)

    for epoch in range(PROJECTION_EPOCHS):
        total_loss = 0.0
        for batch in loader:
            teacher = F.normalize(batch.to(device), dim=1)
            student = F.normalize(projection(teacher), dim=1)
            # Keep caption similarities close to the original frozen BGE-M3.
            loss = F.mse_loss(student @ student.T, teacher @ teacher.T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Projection epoch {epoch + 1}/{PROJECTION_EPOCHS}: loss={total_loss / len(loader):.6f}")

    projection = projection.cpu().requires_grad_(False)
    CAPTION_PROJECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projection.state_dict(), CAPTION_PROJECTION_PATH)
    return projection


def ensure_caption_embeddings(retrain=False):
    encoder = None
    embeddings = {}
    for split in SPLITS:
        rows = load_metadata(split)
        path = caption_embedding_path(split, CAPTION_BASE_EMBEDDING_SIZE)
        expected_shape = (len(rows), CAPTION_BASE_EMBEDDING_SIZE)
        if not path.exists() or np.load(path, mmap_mode="r").shape != expected_shape:
            if encoder is None:
                encoder = load_caption_encoder(projected=False)
            captions = [row["caption"] for row in rows]
            features = encoder.encode(
                captions, batch_size=CAPTION_BATCH_SIZE, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=True,
            ).astype(np.float32)
            np.save(path, features)
        embeddings[split] = np.load(path)
    del encoder

    if TEXT_EMBEDDING_SIZE == CAPTION_BASE_EMBEDDING_SIZE:
        return
    if retrain or not CAPTION_PROJECTION_PATH.exists():
        projection = train_projection(embeddings["train"])
    else:
        projection = nn.Linear(CAPTION_BASE_EMBEDDING_SIZE, TEXT_EMBEDDING_SIZE, bias=False)
        projection.load_state_dict(torch.load(CAPTION_PROJECTION_PATH, map_location="cpu", weights_only=True))

    with torch.no_grad():
        for split in SPLITS:
            features = F.normalize(projection(torch.from_numpy(embeddings[split]).float()), dim=1).numpy()
            np.save(caption_embedding_path(split), features)
            print(f"{split}: saved {features.shape}")


if __name__ == "__main__":
    ensure_caption_embeddings(retrain=True)
