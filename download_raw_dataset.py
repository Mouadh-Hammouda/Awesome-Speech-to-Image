import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils import RAW_DIR, print_progress

SPOKENCOCO_URL = "https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz"
COCO_IMAGES_URL = "http://images.cocodataset.org"

MAX_PAIRS = {"train": 110000, "val": 10000}
IMAGE_DOWNLOAD_THREADS = 16
IMAGE_DOWNLOAD_ATTEMPTS = 3


def read_metadata(archive, archive_entry):
    split = archive_entry.name.removeprefix("SpokenCOCO/SpokenCOCO_").removesuffix(".json")
    metadata_bytes = archive.extractfile(archive_entry).read()
    # Kept for generate_metadata.py, which reads the caption text from it.
    (RAW_DIR / f"SpokenCOCO_{split}.json").write_bytes(metadata_bytes)
    wav_metadata = {}
    for image_entry in json.loads(metadata_bytes)["data"]:
        for caption_number, caption in enumerate(image_entry["captions"]):
            wav_metadata[caption["wav"]] = (split, image_entry["image"], caption_number)
    return wav_metadata


def download_image(url, image_file):
    # Download under a temporary name, so an interrupted download never looks like a finished image.
    partial_file = image_file.with_suffix(".part")
    for attempt in range(1, IMAGE_DOWNLOAD_ATTEMPTS + 1):
        try:
            urllib.request.urlretrieve(url, partial_file)
            partial_file.rename(image_file)
            return
        except OSError:
            if attempt == IMAGE_DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(attempt)


def save_pair(archive, archive_entry, split, coco_image_path, caption_number, executor, image_downloads):
    image_id = Path(coco_image_path).stem[-12:]
    wav_file = RAW_DIR / split / f"{image_id}_{caption_number}.wav"
    image_file = RAW_DIR / split / f"{image_id}.jpg"

    if not image_file.exists() and image_file not in image_downloads:
        image_downloads.add(image_file)
        executor.submit(download_image, f"{COCO_IMAGES_URL}/{coco_image_path}", image_file)
    # Pairs from an earlier, smaller run are kept, so only new pairs are written.
    if not wav_file.exists():
        wav_file.write_bytes(archive.extractfile(archive_entry).read())


def download_pairs(archive, executor):
    wav_metadata = {}
    image_downloads = set()
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

        save_pair(archive, archive_entry, split, coco_image_path, caption_number, executor, image_downloads)
        pair_counts[split] += 1
        print_progress(pair_counts, sum(pair_counts.values()), sum(MAX_PAIRS.values()), start_time)
        if pair_counts == MAX_PAIRS:
            break
    print("\nWaiting for the remaining image downloads...")


def download_missing_images(split):
    # Catches any image whose download failed during the archive pass, without streaming the archive again.
    with open(RAW_DIR / f"SpokenCOCO_{split}.json") as file:
        coco_image_paths = {Path(image_entry["image"]).stem[-12:]: image_entry["image"] for image_entry in json.load(file)["data"]}
    for image_id in sorted({wav_file.stem.rsplit("_", 1)[0] for wav_file in (RAW_DIR / split).glob("*.wav")}):
        image_file = RAW_DIR / split / f"{image_id}.jpg"
        if not image_file.exists():
            print(f"Downloading missing image {image_file}")
            download_image(f"{COCO_IMAGES_URL}/{coco_image_paths[image_id]}", image_file)


def main():
    for split in MAX_PAIRS:
        (RAW_DIR / split).mkdir(parents=True, exist_ok=True)

    # The archive only needs streaming if some split is still short of audio.
    if any(len(list((RAW_DIR / split).glob("*.wav"))) < max_pairs for split, max_pairs in MAX_PAIRS.items()):
        archive = tarfile.open(fileobj=urllib.request.urlopen(SPOKENCOCO_URL), mode="r|gz")
        with ThreadPoolExecutor(IMAGE_DOWNLOAD_THREADS) as executor:
            download_pairs(archive, executor)

    for split in MAX_PAIRS:
        download_missing_images(split)


if __name__ == "__main__":
    main()
