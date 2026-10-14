import numpy as np
import torch
import time
from pathlib import Path
from speechbrain.inference.interfaces import foreign_class

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
SPLITS = ["train", "val"]


def load_sense():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return foreign_class(
        source="LIA-AvignonUniversity/SENSE",
        pymodule_file="custom.py",
        classname="SENSEEncoder",
        run_opts={"device": device},
    )


def print_progress(split, done, total, start_time):
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{split}: {done}/{total}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r")


def encode_split(sense, split):
    wav_files = sorted((RAW_DIR / split).glob("*.wav"))
    embeddings = []
    start_time = time.monotonic()
    for number, wav_file in enumerate(wav_files, start=1):
        embeddings.append(sense.encode_file(str(wav_file)).squeeze(0).cpu().numpy())
        print_progress(split, number, len(wav_files), start_time)
    print()
    return np.stack(embeddings)


def main():
    sense = load_sense()
    for split in SPLITS:
        np.save(OUTPUT_DIR / f"{split}_x.npy", encode_split(sense, split))


if __name__ == "__main__":
    main()
