import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a GUI.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(53)
np.random.seed(53)

EPOCHS = 25
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TEMPERATURE = 0.07
RECALL_KS = (1, 5, 10)
OUTPUT_DIR = Path("results")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"PyTorch {torch.__version__} | device: {device}")


def npy_loader(path: Path):
    """Load an embedding array as a float32 tensor."""
    array = np.load(path, allow_pickle=False)
    return torch.from_numpy(array).float()


class PairedEmbeddingDataset(Dataset):
    def __init__(
        self,
        audio_path: Path,
        image_path: Path,
        metadata_path: Path,
    ):
        self.audio = npy_loader(audio_path)
        self.images = npy_loader(image_path)

        with open(metadata_path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None or "image_id" not in reader.fieldnames:
                raise ValueError(
                    f"{metadata_path} must contain an 'image_id' column"
                )

            rows = list(reader)

        for index, row in enumerate(rows):
            if "row_index" in row and int(row["row_index"]) != index:
                raise ValueError(
                    f"Metadata row_index mismatch at row {index}"
                )

        self.image_ids = torch.tensor(
            [int(row["image_id"]) for row in rows],
            dtype=torch.long,
        )

        if self.audio.ndim != 2 or self.audio.shape[1] != 1024:
            raise ValueError("Expected audio shape (N, 1024)")

        if self.images.ndim != 2 or self.images.shape[1] != 768:
            raise ValueError("Expected image shape (N, 768)")

        if len(self.audio) != len(self.images):
            raise ValueError("Audio and image sample counts must match")

        if len(self.image_ids) != len(self.audio):
            raise ValueError(
                "Metadata and embedding sample counts must match"
            )

        if len(self.audio) == 0:
            raise ValueError("Dataset must not be empty")

        for name, embeddings in (
            ("Audio", self.audio),
            ("Image", self.images),
        ):
            if not torch.isfinite(embeddings).all().item():
                raise ValueError(
                    f"{name} embeddings contain NaN or infinity"
                )

            if (embeddings.norm(dim=1) == 0).any().item():
                raise ValueError(
                    f"{name} embeddings contain zero vectors"
                )

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        return (
            self.audio[index],
            self.images[index],
            self.image_ids[index],
        )


class MLPsound_image(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=768,
        output_dim=768,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_data):
        output = self.model(input_data)
        return F.normalize(output, p=2, dim=-1)


def contrastive_loss(
    predicted,
    images,
    image_ids,
    temperature=TEMPERATURE,
):
    """Contrastive loss with all matching image IDs as positives."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    predicted = F.normalize(predicted, p=2, dim=-1)
    images = F.normalize(images, p=2, dim=-1)

    scores = (predicted @ images.T) / temperature

    positive_mask = image_ids[:, None] == image_ids[None, :]

    positive_scores = scores.masked_fill(
        ~positive_mask,
        float("-inf"),
    )

    log_positive = torch.logsumexp(positive_scores, dim=1)
    log_all = torch.logsumexp(scores, dim=1)

    return (log_all - log_positive).mean()


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for audio, images, image_ids in loader:
            audio = audio.to(device)
            images = images.to(device)
            image_ids = image_ids.to(device)

            predicted = model(audio)
            loss = contrastive_loss(predicted, images, image_ids)

            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

    return total_loss / total_samples


def build_image_gallery(dataset, device):
    """Keep one image embedding per unique validation image ID."""
    first_row_by_id = {}

    for row_index, image_id in enumerate(dataset.image_ids.tolist()):
        if image_id not in first_row_by_id:
            first_row_by_id[image_id] = row_index

    gallery_rows = list(first_row_by_id.values())

    gallery_images = dataset.images[gallery_rows].to(device)
    gallery_ids = dataset.image_ids[gallery_rows].to(device)

    gallery_images = F.normalize(gallery_images, p=2, dim=-1)

    return gallery_images, gallery_ids


def evaluate_recall(
    model,
    loader,
    gallery_images,
    gallery_ids,
    device,
    ks=RECALL_KS,
):
    """Evaluate every query against the full unique-image gallery."""
    gallery_size = len(gallery_ids)

    if not ks or any(k < 1 or k > gallery_size for k in ks):
        raise ValueError(
            f"Recall K values must be between 1 and {gallery_size}"
        )

    model.eval()
    correct_counts = {k: 0 for k in ks}
    total_queries = 0

    with torch.no_grad():
        for audio, _, target_ids in loader:
            audio = audio.to(device)
            target_ids = target_ids.to(device)

            predicted = F.normalize(model(audio), p=2, dim=-1)

            # Shape: (audio batch size, number of unique images).
            similarities = predicted @ gallery_images.T

            top_indices = similarities.topk(
                k=max(ks),
                dim=1,
            ).indices

            retrieved_ids = gallery_ids[top_indices]

            for k in ks:
                matches = (
                    retrieved_ids[:, :k] == target_ids[:, None]
                )

                correct_counts[k] += (
                    matches.any(dim=1).sum().item()
                )

            total_queries += audio.size(0)

    if total_queries == 0:
        raise ValueError("No queries were evaluated")

    return {
        f"recall@{k}": 100.0 * correct_counts[k] / total_queries
        for k in ks
    }


def save_history_and_plots(history, output_dir):
    """Save metric values and plots; does not save the model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train"]) + 1))

    # Save values for later analysis.
    csv_path = output_dir / "training_history.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *history.keys()])

        for index, epoch in enumerate(epochs):
            writer.writerow(
                [epoch] + [values[index] for values in history.values()]
            )

    # Plot training and validation loss.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs, history["train"],
        marker="o", label="Training loss",
    )
    ax.plot(
        epochs, history["valid"],
        marker="o", label="Validation loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Batch contrastive loss")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    loss_path = output_dir / "loss_curves.png"
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)

    # Plot full-gallery validation recall.
    fig, ax = plt.subplots(figsize=(8, 5))

    for k in RECALL_KS:
        ax.plot(
            epochs,
            history[f"recall@{k}"],
            marker="o",
            label=f"Recall@{k}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Successful queries (%)")
    ax.set_title("Validation retrieval — full unique-image gallery")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    recall_path = output_dir / "recall_curves.png"
    fig.savefig(recall_path, dpi=200)
    plt.close(fig)

    print(f"History saved to: {csv_path.resolve()}")
    print(f"Loss plot saved to: {loss_path.resolve()}")
    print(f"Recall plot saved to: {recall_path.resolve()}")


def main(
    train_audio_path: Path,
    train_images_path: Path,
    train_metadata_path: Path,
    val_audio_path: Path,
    val_images_path: Path,
    val_metadata_path: Path,
):
    train_dataset = PairedEmbeddingDataset(
        train_audio_path,
        train_images_path,
        train_metadata_path,
    )

    valid_dataset = PairedEmbeddingDataset(
        val_audio_path,
        val_images_path,
        val_metadata_path,
    )

    train_ids = set(train_dataset.image_ids.tolist())
    val_ids = set(valid_dataset.image_ids.tolist())

    if train_ids & val_ids:
        raise ValueError("Training and validation share image IDs")

    if len(train_ids) < 2:
        raise ValueError("Training needs at least two distinct images")

    if len(val_ids) < max(RECALL_KS):
        raise ValueError(
            f"Validation needs at least {max(RECALL_KS)} unique images "
            f"to calculate Recall@{max(RECALL_KS)}"
        )

    print("Training pairs:", len(train_dataset))
    print("Validation pairs:", len(valid_dataset))
    print("Unique training images:", len(train_ids))
    print("Unique validation images:", len(val_ids))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Build once: frozen image embeddings do not change during training.
    gallery_images, gallery_ids = build_image_gallery(
        valid_dataset, device
    )

    model = MLPsound_image().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(model)
    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {parameter_count:,}")

    history = {
        "train": [],
        "valid": [],
        "recall@1": [],
        "recall@5": [],
        "recall@10": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for audio, images, image_ids in train_loader:
            audio = audio.to(device)
            images = images.to(device)
            image_ids = image_ids.to(device)

            optimizer.zero_grad()

            predicted = model(audio)
            loss = contrastive_loss(predicted, images, image_ids)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * audio.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        valid_loss = evaluate_loss(model, valid_loader, device)

        recalls = evaluate_recall(
            model,
            valid_loader,
            gallery_images,
            gallery_ids,
            device,
        )

        history["train"].append(train_loss)
        history["valid"].append(valid_loss)

        for name, value in recalls.items():
            history[name].append(value)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"valid loss: {valid_loss:.4f} | "
            f"R@1: {recalls['recall@1']:.2f}% | "
            f"R@5: {recalls['recall@5']:.2f}% | "
            f"R@10: {recalls['recall@10']:.2f}%"
        )

    best_index = int(np.argmax(history["recall@1"]))
    print(
        f"\nBest validation Recall@1: "
        f"{history['recall@1'][best_index]:.2f}% "
        f"at epoch {best_index + 1}"
    )
    print("Returned model contains the final epoch's weights, not the best epoch's.")

    return model, history


if __name__ == "__main__":
    model, history = main(
        train_audio_path=Path("data/train_x.npy"),
        train_images_path=Path("data/train_y.npy"),
        train_metadata_path=Path("data/train_meta.csv"),
        val_audio_path=Path("data/val_x.npy"),
        val_images_path=Path("data/val_y.npy"),
        val_metadata_path=Path("data/val_meta.csv"),
    )

    save_history_and_plots(history, OUTPUT_DIR)
