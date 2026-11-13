import random
import torch
import torch.nn.functional as F
from caption_data import CaptionSplit, augment, centre_crop
from caption_models import CaptionTextEncoder, ImageEncoderCNN, model_file
from utils import get_device

EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TEMPERATURE = 0.05
RECALL_KS = (1, 5, 10)
# Names from caption_data.CLASSES to train and validate on, e.g. ("CAT", "DOG"); None trains on every image.
TRAIN_CLASSES = ("CAT", "DOG")

torch.manual_seed(53)
random.seed(53)
device = get_device()


def train_epoch(image_encoder, text_encoder, optimizer, data):
    image_encoder.train()
    text_encoder.train()
    total_loss = 0.0
    for images in torch.randperm(len(data.images)).split(BATCH_SIZE):
        # One random caption per image, so each image's only positive is its own caption.
        captions = data.captions[[random.choice(data.image_captions[image]) for image in images.tolist()]].to(device)
        scores = image_encoder(augment(data.images[images]).to(device)) @ text_encoder(captions).T / TEMPERATURE
        targets = torch.arange(len(images), device=device)
        loss = (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(data.images)


def recall(scores, is_match):
    hits = is_match.gather(1, scores.topk(max(RECALL_KS), dim=1).indices)
    return [100.0 * hits[:, :k].any(dim=1).float().mean().item() for k in RECALL_KS]


@torch.no_grad()
def evaluate(image_encoder, text_encoder, data):
    image_encoder.eval()
    text_encoder.eval()
    image_embeddings = torch.cat([image_encoder(centre_crop(images).to(device)) for images in data.images.split(BATCH_SIZE)])
    caption_embeddings = torch.cat([text_encoder(captions.to(device)) for captions in data.captions.split(BATCH_SIZE)])
    is_match = data.caption_images.to(device)[:, None] == torch.arange(len(image_embeddings), device=device)
    # A hit if the caption's own image is among its top-k closest images.
    scores = caption_embeddings @ image_embeddings.T
    return recall(scores, is_match)


def main():
    output_file = model_file(TRAIN_CLASSES)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print("Loading images...", flush=True)
    train_data = CaptionSplit("train", class_names=TRAIN_CLASSES)
    val_data = CaptionSplit("val", train_data.vocabulary, TRAIN_CLASSES)
    image_encoder = ImageEncoderCNN().to(device)
    text_encoder = CaptionTextEncoder(len(train_data.vocabulary) + 2).to(device)
    optimizer = torch.optim.Adam([*image_encoder.parameters(), *text_encoder.parameters()], lr=LEARNING_RATE)
    print(f"classes: {', '.join(TRAIN_CLASSES) if TRAIN_CLASSES else 'all'} | train images: {len(train_data.images):,} | val images: {len(val_data.images):,}", flush=True)
    print(f"vocabulary: {len(train_data.vocabulary):,} words", flush=True)

    best_recall = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(image_encoder, text_encoder, optimizer, train_data)
        recalls = evaluate(image_encoder, text_encoder, val_data)
        print(f"epoch {epoch} | loss {train_loss:.3f} | text->image R@1/5/10 " + "/".join(f"{value:.1f}" for value in recalls), flush=True)
        if recalls[-1] > best_recall:
            best_recall = recalls[-1]
            torch.save({"image_encoder": image_encoder.state_dict(), "text_encoder": text_encoder.state_dict(), "vocabulary": train_data.vocabulary}, output_file)
            print(f"saved best model to {output_file} (R@10 {best_recall:.1f})", flush=True)


if __name__ == "__main__":
    main()
