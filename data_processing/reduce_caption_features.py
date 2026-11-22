"""Reduces the stored BGE-M3 caption vectors to EMBEDDING_SIZE with a PCA fitted on the train captions."""
import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 data_processing/reduce_caption_features.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from common import EMBEDDING_SIZE, FEATURES_DIR, SPLITS, caption_file
from models.caption_text_encoder import PCA_FILE, project


def principal_directions(vectors, size):
    # The vectors are already centred, so these are the top eigenvectors of their second moment, strongest first.
    vectors = vectors.double()
    directions = torch.linalg.eigh(vectors.T @ vectors).eigenvectors[:, -size:].flip(1)
    # An eigenvector's sign is arbitrary: make its largest entry positive, so every machine gets the same space.
    largest = directions.gather(0, directions.abs().argmax(dim=0, keepdim=True))
    return (directions * largest.sign()).float()


def main():
    full_files = {split: FEATURES_DIR / f"{split}_text.npy" for split in SPLITS}
    for file in full_files.values():
        if not file.exists():
            raise SystemExit(f"{file} is missing: run python3 data_processing/extract_caption_features.py first.")
    captions = {split: torch.from_numpy(np.load(file)).float() for split, file in full_files.items()}

    directions = principal_directions(captions["train"], EMBEDDING_SIZE)
    np.save(PCA_FILE, directions.numpy())
    print(f"Captions reduced to {EMBEDDING_SIZE}-d with PCA")
    print("  split  captions  variance kept  file")
    for split, vectors in captions.items():
        kept = (vectors @ directions).square().sum() / vectors.square().sum()
        reduced = project(vectors, directions)
        np.save(caption_file(split), reduced.numpy())
        print(f"  {split:<5}{len(reduced):>10,}{100 * kept.item():>14.1f}%  {caption_file(split)}")


if __name__ == "__main__":
    main()
