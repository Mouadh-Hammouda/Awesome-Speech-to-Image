import json
import tarfile
import time
import urllib.request
from pathlib import Path

SPOKENCOCO_URL = "https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz"
COCO_IMAGES_URL = "http://images.cocodataset.org"

MAX_PAIRS = {"train": 25000, "val": 10000}
DATA_DIR = Path("data/raw")


def read_metadata(archive, archive_entry):
    split = archive_entry.name.removeprefix("SpokenCOCO/SpokenCOCO_").removesuffix(".json")
    wav_metadata = {}
    for image_entry in json.load(archive.extractfile(archive_entry))["data"]:
        for caption_number, caption in enumerate(image_entry["captions"]):
            wav_metadata[caption["wav"]] = (split, image_entry["image"], caption_number)
    return wav_metadata


def save_pair(archive, archive_entry, split, coco_image_path, caption_number):
    image_id = Path(coco_image_path).stem[-12:]
    wav_file = DATA_DIR / split / f"{image_id}_{caption_number}.wav"
    image_file = DATA_DIR / split / f"{image_id}.jpg"

    if not image_file.exists():
        urllib.request.urlretrieve(f"{COCO_IMAGES_URL}/{coco_image_path}", image_file)
    wav_file.write_bytes(archive.extractfile(archive_entry).read())


def print_progress(pair_counts, start_time):
    done, total = sum(pair_counts.values()), sum(MAX_PAIRS.values())
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(f"{pair_counts}  {elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ", end="\r")


def download_pairs(archive):
    wav_metadata = {}
    pair_counts = dict.fromkeys(MAX_PAIRS, 0)
    start_time = time.monotonic()

    for archive_entry in archive:
        archive_path = archive_entry.name.removeprefix("SpokenCOCO/")

        if archive_path.endswith(".json"):
            wav_metadata |= read_metadata(archive, archive_entry)
            continue

        if archive_path not in wav_metadata:
            continue

        split, coco_image_path, caption_number = wav_metadata[archive_path]
        if pair_counts[split] == MAX_PAIRS[split]:
            continue

        save_pair(archive, archive_entry, split, coco_image_path, caption_number)
        pair_counts[split] += 1
        print_progress(pair_counts, start_time)
        if pair_counts == MAX_PAIRS:
            break


def main():
    for split in MAX_PAIRS:
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    archive = tarfile.open(fileobj=urllib.request.urlopen(SPOKENCOCO_URL), mode="r|gz")
    download_pairs(archive)


if __name__ == "__main__":
    main()
