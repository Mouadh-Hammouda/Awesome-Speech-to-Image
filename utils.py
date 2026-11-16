import time
import torch
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
SPLITS = ["train", "val"]
_image_files = {}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_splits():
    return [split for split in SPLITS if (DATA_DIR / f"{split}_meta.csv").exists()]


def image_file(image_id):
    # Earlier downloads put images in different folders under data/raw/, so all of them are searched.
    if not _image_files:
        _image_files.update({path.stem: path for path in RAW_DIR.glob("*/*.jpg")})
    return _image_files[image_id]


def print_progress(label, done, total, start_time):
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{label}: {done}/{total}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r")
