"""Shared settings and helpers for data preparation, training, and retrieval."""

import time
from pathlib import Path

import torch

DATA_DIR = Path("data")
RAW_DIR = Path("data/raw")
# Resized images and caption vectors, prepared once by the scripts in data_processing/.
FEATURES_DIR = DATA_DIR / "features"
SPLITS = ["train", "val"]
RESULTS_DIR = Path("results/image_to_caption")

# Images are cached larger than the CNN's input, so that a random crop still has detail.
CACHE_SIZE = 224
IMAGE_SIZE = 128
# The pretrained text model that defines the caption space.
TEXT_MODEL = "BAAI/bge-m3"
# The latent space shared by images and captions: the CNN outputs this size, and BGE-M3's 1024 numbers are reduced to it by PCA.
EMBEDDING_SIZE = 256

# Class name -> caption words that put an image in that class (see data_processing/caption_data.py for how captions vote).
# These are only for diagnostics; training never sees them.
CLASSES = {
    "CAT": {"CAT", "CATS", "KITTEN", "KITTENS", "KITTY"},
    "DOG": {"DOG", "DOGS", "PUPPY", "PUPPIES"},
    "PIZZA": {"PIZZA", "PIZZAS"},
    "CAR": {"CAR", "CARS", "AUTOMOBILE", "AUTOMOBILES", "TAXI", "TAXIS", "SEDAN", "SEDANS", "SUV", "SUVS"},
    "BUILDING": {"BUILDING", "BUILDINGS", "HOUSE", "HOUSES", "SKYSCRAPER", "SKYSCRAPERS", "TOWER", "TOWERS"},
    "PLANE": {"PLANE", "PLANES", "AIRPLANE", "AIRPLANES", "AEROPLANE", "AEROPLANES", "AIRCRAFT", "AIRLINER", "AIRLINERS"},
    "COMPUTER": {"PC", "COMPUTER", "COMPUTERS", "LAPTOP", "LAPTOPS", "DESKTOP", "MACBOOK", "IMAC"},
    "PEOPLE": {
        "PERSON", "PEOPLE", "MAN", "MEN", "WOMAN", "WOMEN", "BOY", "BOYS", "GIRL", "GIRLS",
        "CHILD", "CHILDREN", "BABY", "BABIES", "KID", "KIDS", "TODDLER", "TODDLERS",
        "ADULT", "ADULTS", "CROWD", "CROWDS", "HUMAN", "HUMANS", "GENTLEMAN", "GENTLEMEN", "LADY", "LADIES",
    },
}


def model_file():
    return RESULTS_DIR / "model.pt"


def caption_file(split):
    # The size is in the name, so these never mix with the full BGE-M3 vectors in {split}_text.npy.
    return FEATURES_DIR / f"{split}_text_{EMBEDDING_SIZE}.npy"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_progress(label, done, total, start_time):
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{label}: {done}/{total}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r", flush=True)
