import csv

from config import (
    CAPTION_BASE_EMBEDDING_SIZE, DATA_DIR, IMAGE_MEAN, IMAGE_SIZE,
    IMAGE_STD, RAW_DIR, TEXT_EMBEDDING_SIZE,
)


def load_metadata(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def caption_embedding_path(split, dimensions=TEXT_EMBEDDING_SIZE):
    suffix = "" if dimensions == CAPTION_BASE_EMBEDDING_SIZE else f"_{dimensions}"
    return DATA_DIR / f"{split}_bge_m3{suffix}.npy"


def get_image_path(split, image_id):
    return RAW_DIR / split / f"{image_id}.jpg"


def get_image_transform():
    # Caption encoding and downloading do not need torchvision to use utils.
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def unique_image_indices(rows):
    first_index = {}
    for index, row in enumerate(rows):
        first_index.setdefault(row["image_id"], index)
    return list(first_index.values())


def load_image_gallery(split):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from config import BATCH_SIZE, CHECKPOINT_PATH
    from image_cnn import ImageCNN
    from image_dataset import ImageCaptionDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    dataset = ImageCaptionDataset(split)
    loader = DataLoader(Subset(dataset, unique_image_indices(dataset.rows)), batch_size=BATCH_SIZE)
    features = []
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc=f"Encode {split} images", unit="batch"):
            features.append(model(images.to(device)).cpu())
    return dataset, torch.cat(features)
