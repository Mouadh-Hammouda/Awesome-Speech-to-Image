import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
SPLITS = ["train", "val"]


def write_metadata(split):
    # Same sorted wav order as extract_sense_features.py, so row i matches row i of {split}_x.npy.
    wav_files = sorted((RAW_DIR / split).glob("*.wav"))
    with open(OUTPUT_DIR / f"{split}_meta.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "audio_filename", "image_id"])
        for row, wav_file in enumerate(wav_files):
            writer.writerow([row, wav_file.name, wav_file.stem.rsplit("_", 1)[0]])


def main():
    for split in SPLITS:
        write_metadata(split)


if __name__ == "__main__":
    main()
