import csv
import json
from pathlib import Path

from config import DATA_DIR, RAW_DIR, SPLITS


def match_captions_to_image(split):
    with open(RAW_DIR / f"SpokenCOCO_{split}.json", encoding="utf-8") as file:
        items = json.load(file)["data"]

    captions_by_image = {}
    for entry in items:
        captions_by_image[Path(entry["image"]).stem.rsplit("_", 1)[-1]] = entry["captions"]

    return captions_by_image


def write_metadata(split):
    # Keep WAVs sorted to preserve alignment with the existing embeddings.
    wav_files = sorted((RAW_DIR / split).glob("*.wav"))
    captions_by_image = match_captions_to_image(split)

    rows = []
    for row, wav_file in enumerate(wav_files):
        image_id, caption_index = wav_file.stem.rsplit("_", 1)
        caption = captions_by_image[image_id][int(caption_index)]["text"]
        rows.append([row, wav_file.name, image_id, caption])

    with open(DATA_DIR / f"{split}_meta.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "audio_filename", "image_id", "caption"])
        writer.writerows(rows)


def main():
    for split in SPLITS:
        write_metadata(split)


if __name__ == "__main__":
    main()
