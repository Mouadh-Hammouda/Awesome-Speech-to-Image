"""Shared encoder loading and image embeddings for search and diagnostics."""

import torch

from data_processing.caption_data import centre_crop, load_images
from common import RAW_DIR, get_device, model_file
from models.caption_text_encoder import CaptionTextEncoder
from models.image_cnn import ImageEncoderCNN

BATCH_SIZE = 128

device = get_device()


def load_encoders(class_names=None):
    checkpoint = torch.load(model_file(class_names), map_location=device)
    vocabulary = checkpoint["vocabulary"]
    image_encoder = ImageEncoderCNN().to(device)
    text_encoder = CaptionTextEncoder(len(vocabulary) + 2).to(device)
    image_encoder.load_state_dict(checkpoint["image_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    return image_encoder.eval(), text_encoder.eval(), vocabulary


@torch.no_grad()
def embed_split(image_encoder, split, class_names=None):
    trained_file = model_file(class_names)
    embeddings_file = trained_file.parent / f"{split}_image_embeddings.pt"
    # Saved embeddings are out of date once the model has been retrained.
    if embeddings_file.exists() and embeddings_file.stat().st_mtime > trained_file.stat().st_mtime:
        saved = torch.load(embeddings_file, map_location=device)
        image_ids, image_embeddings = saved["image_ids"], saved["image_embeddings"]
    else:
        image_ids = sorted(image_file.stem for image_file in (RAW_DIR / split).glob("*.jpg"))
        images = load_images(split, image_ids)
        image_embeddings = torch.cat([image_encoder(centre_crop(batch).to(device)) for batch in images.split(BATCH_SIZE)])
        torch.save({"image_ids": image_ids, "image_embeddings": image_embeddings.cpu()}, embeddings_file)
    return [RAW_DIR / split / f"{image_id}.jpg" for image_id in image_ids], image_embeddings
