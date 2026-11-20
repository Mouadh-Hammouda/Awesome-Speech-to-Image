"""Shared settings and helpers for data preparation, training, and retrieval."""

import time
from pathlib import Path

import torch

DATA_DIR = Path("data")
RAW_DIR = Path("data/raw")
SPLITS = ["train", "val"]
RESULTS_DIR = Path("results/image_to_caption")

LOAD_SIZE = 140
IMAGE_SIZE = 128
MIN_WORD_COUNT = 5
MAX_CAPTION_WORDS = 40
PADDING, UNKNOWN = 0, 1
EMBEDDING_SIZE = 256
TEMPERATURE = 0.05

# Class name -> caption words that put an image in that class.
# class_of() uses the first match; keep PEOPLE last so objects take priority.
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


def model_file(class_names=None):
    # A model trained on only some classes gets its own folder, e.g. cat_dog/, so it never overwrites another.
    if class_names is None:
        return RESULTS_DIR / "model.pt"
    return RESULTS_DIR / "_".join(sorted(name.lower() for name in class_names)) / "model.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_progress(label, done, total, start_time):
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{label}: {done}/{total}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r")
