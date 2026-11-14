import csv

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from config import (
    BATCH_SIZE, CHECKPOINT_PATH, EPOCHS, HISTORY_PATH,
    LEARNING_RATE, RESULTS_DIR, SEED,
)
from image_cnn import ImageCNN
from image_dataset import ImageCaptionDataset
from encode_captions import ensure_caption_embeddings
from utils import unique_image_indices


def alignment_loss(image_embeddings, text_embeddings):
    similarities = (image_embeddings * text_embeddings).sum(dim=1)
    return (1 - similarities).mean()


@torch.no_grad()
def evaluate_recall_at_1(model, dataset, device):
    model.eval()

    loader = DataLoader(
        Subset(dataset, unique_image_indices(dataset.rows)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    caption_gallery = dataset.targets.to(device)
    caption_ids = [row["image_id"] for row in dataset.rows]

    correct = 0
    total = 0

    progress = tqdm(loader, desc="Validation", unit="batch", dynamic_ncols=True)
    for images, _, image_ids in progress:
        predictions = model(images.to(device))
        scores = predictions @ caption_gallery.T
        best_indices = scores.argmax(dim=1).cpu().tolist()

        for index, image_id in zip(best_indices, image_ids):
            correct += int(caption_ids[index] == image_id)
            total += 1

        progress.set_postfix(recall=f"{correct / total:.2%}")

    return correct / total


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_caption_embeddings()
    train_dataset = ImageCaptionDataset("train")
    val_dataset = ImageCaptionDataset("val")

    train_ids = {row["image_id"] for row in train_dataset.rows}
    val_ids = {row["image_id"] for row in val_dataset.rows}

    if train_ids & val_ids:
        raise ValueError("Training and validation share images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ImageCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} train",
            unit="batch", dynamic_ncols=True,
        )
        for images, targets, _ in progress:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = alignment_loss(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")

        average_loss = total_loss / len(train_dataset)
        recall = evaluate_recall_at_1(model, val_dataset, device)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={average_loss:.4f}, "
            f"validation Recall@1={recall:.2%}"
        )

        history.append([epoch + 1, average_loss, recall])
        with open(HISTORY_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "loss", "recall_at_1"])
            writer.writerows(history)

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
