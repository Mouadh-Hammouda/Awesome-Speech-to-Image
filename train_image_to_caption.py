import random
import torch
import torch.nn.functional as F
from caption_data import CaptionSplit, augment, centre_crop
from caption_models import CaptionTextEncoder, ImageEncoderCNN, model_file
from utils import get_device

EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
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
    # Decays the learning rate from LEARNING_RATE towards zero along a cosine curve, stepped once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"classes: {', '.join(TRAIN_CLASSES) if TRAIN_CLASSES else 'all'} | train images: {len(train_data.images):,} | val images: {len(val_data.images):,}", flush=True)
    print(f"vocabulary: {len(train_data.vocabulary):,} words", flush=True)

    # The latest training state, saved every epoch so an interrupted run picks up where it left off.
    checkpoint_file = output_file.with_name("checkpoint.pt")
    best_recall = -1.0
    start_epoch = 1
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        image_encoder.load_state_dict(checkpoint["image_encoder"])
        text_encoder.load_state_dict(checkpoint["text_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_recall = checkpoint["best_recall"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"resuming from {checkpoint_file} at epoch {start_epoch} (best R@10 {best_recall:.1f})", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        learning_rate = scheduler.get_last_lr()[0]
        train_loss = train_epoch(image_encoder, text_encoder, optimizer, train_data)
        scheduler.step()
        recalls = evaluate(image_encoder, text_encoder, val_data)
        print(f"epoch {epoch} | lr {learning_rate:.2e} | loss {train_loss:.3f} | text->image R@1/5/10 " + "/".join(f"{value:.1f}" for value in recalls), flush=True)
        if recalls[-1] > best_recall:
            best_recall = recalls[-1]
            torch.save({"image_encoder": image_encoder.state_dict(), "text_encoder": text_encoder.state_dict(), "vocabulary": train_data.vocabulary}, output_file)
            print(f"saved best model to {output_file} (R@10 {best_recall:.1f})", flush=True)
        torch.save({"image_encoder": image_encoder.state_dict(), "text_encoder": text_encoder.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_recall": best_recall}, checkpoint_file)


if __name__ == "__main__":
    main()
