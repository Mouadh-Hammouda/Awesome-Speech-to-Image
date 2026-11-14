import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config import (
    COCO_IMAGES_URL, IMAGE_DOWNLOAD_ATTEMPTS, IMAGE_DOWNLOAD_THREADS,
    MAX_PAIRS, RAW_DIR, SPLITS, SPOKENCOCO_URL,
)
from utils import get_image_path


def read_metadata(split):
    with open(RAW_DIR / f"SpokenCOCO_{split}.json", encoding="utf-8") as file:
        images = json.load(file)["data"]
    wav_metadata = {}
    for image_entry in images:
        for caption_number, caption in enumerate(image_entry["captions"]):
            wav_metadata[caption["wav"]] = (split, image_entry["image"], caption_number)
    return wav_metadata


def get_wav_path(split, coco_image_path, caption_number):
    image_id = Path(coco_image_path).stem[-12:]
    return RAW_DIR / split / f"{image_id}_{caption_number}.wav"


def get_pair_limit(split, available):
    value = str(MAX_PAIRS[split]).strip()
    if value.lower() == "max":
        return available
    limit = int(value)
    if limit < 0:
        raise ValueError("Pair limits must be non-negative numbers or 'Max'")
    return min(limit, available)


def download_image(url, image_file):
    partial_file = image_file.with_suffix(".part")
    for attempt in range(1, IMAGE_DOWNLOAD_ATTEMPTS + 1):
        try:
            urllib.request.urlretrieve(url, partial_file)
            partial_file.replace(image_file)
            return
        except OSError:
            if attempt == IMAGE_DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(attempt)


def queue_image(split, coco_image_path, executor, image_downloads):
    image_file = get_image_path(split, Path(coco_image_path).stem[-12:])
    if not image_file.exists() and image_file not in image_downloads:
        image_downloads[image_file] = executor.submit(
            download_image, f"{COCO_IMAGES_URL}/{coco_image_path}", image_file
        )


def save_pair(archive, archive_entry, split, coco_image_path, caption_number, executor, image_downloads):
    queue_image(split, coco_image_path, executor, image_downloads)
    wav_file = get_wav_path(split, coco_image_path, caption_number)
    if not wav_file.exists():
        partial_file = wav_file.with_suffix(".part")
        partial_file.write_bytes(archive.extractfile(archive_entry).read())
        partial_file.replace(wav_file)


def print_progress(pair_counts, pair_totals, start_time):
    done, total = sum(pair_counts.values()), sum(pair_totals.values())
    elapsed_minutes = (time.monotonic() - start_time) / 60
    remaining_minutes = elapsed_minutes / done * (total - done)
    print(
        f"{pair_counts}  {done:,}/{total:,} pairs  "
        f"{elapsed_minutes:.0f} min elapsed, ~{remaining_minutes:.0f} min left   ",
        end="\r", flush=True,
    )


def download_pairs(archive, executor, image_downloads, wav_metadata, pair_counts, pair_totals):
    metadata_files = {f"SpokenCOCO_{split}.json": split for split in SPLITS}
    known_splits = {split for split in SPLITS if (RAW_DIR / f"SpokenCOCO_{split}.json").exists()}
    start_time = time.monotonic()

    for archive_entry in archive:
        archive_path = archive_entry.name.removeprefix("SpokenCOCO/")

        if archive_path in metadata_files:
            split = metadata_files[archive_path]
            (RAW_DIR / archive_path).write_bytes(archive.extractfile(archive_entry).read())
            metadata = read_metadata(split)
            wav_metadata.update(metadata)
            pair_totals[split] = get_pair_limit(split, len(metadata))
            existing = sum(get_wav_path(*pair).exists() for pair in metadata.values())
            pair_counts[split] = min(existing, pair_totals[split])
            known_splits.add(split)
        elif archive_path in wav_metadata:
            split, coco_image_path, caption_number = wav_metadata[archive_path]
            if pair_counts[split] < pair_totals[split] and not get_wav_path(split, coco_image_path, caption_number).exists():
                save_pair(archive, archive_entry, split, coco_image_path, caption_number, executor, image_downloads)
                pair_counts[split] += 1
                if sum(pair_counts.values()) % 100 == 0:
                    print_progress(pair_counts, pair_totals, start_time)

        if len(known_splits) == len(SPLITS) and all(pair_counts[split] >= pair_totals[split] for split in SPLITS):
            break

    if sum(pair_counts.values()):
        print_progress(pair_counts, pair_totals, start_time)
        print()


def download_missing_images(wav_metadata, executor, image_downloads):
    image_paths = {
        (split, Path(coco_path).stem[-12:]): coco_path
        for split, coco_path, _ in wav_metadata.values()
    }
    for split in SPLITS:
        image_ids = {wav.stem.rsplit("_", 1)[0] for wav in (RAW_DIR / split).glob("*.wav")}
        for image_id in image_ids:
            if (split, image_id) in image_paths:
                queue_image(split, image_paths[split, image_id], executor, image_downloads)


def main():
    wav_metadata = {}
    pair_counts = dict.fromkeys(SPLITS, 0)
    pair_totals = dict.fromkeys(SPLITS, 0)
    for split in SPLITS:
        (RAW_DIR / split).mkdir(parents=True, exist_ok=True)
        if (RAW_DIR / f"SpokenCOCO_{split}.json").exists():
            metadata = read_metadata(split)
            wav_metadata.update(metadata)
            pair_totals[split] = get_pair_limit(split, len(metadata))
            existing = sum(get_wav_path(*pair).exists() for pair in metadata.values())
            pair_counts[split] = min(existing, pair_totals[split])

    missing_metadata = any(not (RAW_DIR / f"SpokenCOCO_{split}.json").exists() for split in SPLITS)
    missing_audio = any(pair_counts[split] < pair_totals[split] for split in SPLITS)
    with ThreadPoolExecutor(IMAGE_DOWNLOAD_THREADS) as executor:
        image_downloads = {}
        if missing_metadata or missing_audio:
            with urllib.request.urlopen(SPOKENCOCO_URL) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    download_pairs(archive, executor, image_downloads, wav_metadata, pair_counts, pair_totals)
        else:
            print("Configured recording limits are already satisfied.")

        download_missing_images(wav_metadata, executor, image_downloads)
        for future in tqdm(as_completed(image_downloads.values()), total=len(image_downloads), desc="Images", unit="image"):
            future.result()
    print("Download complete.")


if __name__ == "__main__":
    main()
