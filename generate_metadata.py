import csv
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
SPLITS = ["train", "val"]


def read_captions(split):
    # Same (image_id, caption_number) naming as download_raw_dataset.py.
    with open(RAW_DIR / f"SpokenCOCO_{split}.json") as file:
        image_entries = json.load(file)["data"]
    return {
        f"{Path(image_entry['image']).stem[-12:]}_{caption_number}": caption["text"]
        for image_entry in image_entries
        for caption_number, caption in enumerate(image_entry["captions"])
    }


def write_metadata(split):
    captions = read_captions(split)
    # Same sorted wav order as extract_sense_features.py, so row i matches row i of {split}_x.npy.
    wav_files = sorted((RAW_DIR / split).glob("*.wav"))
    with open(OUTPUT_DIR / f"{split}_meta.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "audio_filename", "image_id", "caption"])
        for row, wav_file in enumerate(wav_files):
            writer.writerow([row, wav_file.name, wav_file.stem.rsplit("_", 1)[0], captions[wav_file.stem]])


def main():
    for split in SPLITS:
        write_metadata(split)


if __name__ == "__main__":
    main()
