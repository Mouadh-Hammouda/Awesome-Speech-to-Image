import csv
import re

import matplotlib.pyplot as plt
import numpy as np

from config import (
    DATA_DIR, DEBUG_CLASSES, DEBUG_DIR, DEBUG_FEATURES, DEBUG_SAMPLES_PER_CLASS,
    DEBUG_SPLIT, HISTORY_PATH, SEED,
)
from utils import load_image_gallery, load_metadata


def caption_groups(rows):
    """Use caption keywords as approximate labels, excluding mixed groups."""
    patterns = [re.compile(rf"\b{re.escape(name)}s?\b", re.IGNORECASE) for name in DEBUG_CLASSES]
    matches = {}
    for row in rows:
        labels = matches.setdefault(row["image_id"], set())
        for label, pattern in enumerate(patterns):
            if pattern.search(row["caption"]):
                labels.add(label)
    return {image_id: next(iter(labels)) for image_id, labels in matches.items() if len(labels) == 1}


def training_class_prototypes():
    rows = load_metadata("train")
    targets = np.load(DATA_DIR / "train_bge_m3.npy", mmap_mode="r")
    groups = caption_groups(rows)
    prototypes = []
    for label, name in enumerate(DEBUG_CLASSES):
        indices = [i for i, row in enumerate(rows) if groups.get(row["image_id"]) == label]
        if not indices:
            raise ValueError(f"No training captions for '{name}'. Choose another DEBUG_CLASSES keyword.")
        prototype = targets[indices].mean(axis=0)
        prototypes.append(prototype / np.linalg.norm(prototype))
    return np.stack(prototypes)


def select_samples(dataset):
    groups = caption_groups(dataset.rows)
    rng = np.random.default_rng(SEED)
    indices, labels = [], []
    for label, name in enumerate(DEBUG_CLASSES):
        candidates = [i for i, image_id in enumerate(dataset.image_ids) if groups.get(image_id) == label]
        if not candidates:
            raise ValueError(f"No {DEBUG_SPLIT} images for '{name}'. Choose another DEBUG_CLASSES keyword.")
        selected = rng.choice(candidates, min(len(candidates), DEBUG_SAMPLES_PER_CLASS), replace=False)
        indices.extend(selected)
        labels.extend([label] * len(selected))
        print(f"{name}: {len(selected)} images")
    return np.array(indices), np.array(labels)


def average_captions(dataset, indices):
    image_ids = [dataset.image_ids[index] for index in indices]
    captions = {image_id: [] for image_id in image_ids}
    for row, target in zip(dataset.rows, dataset.targets.numpy()):
        if row["image_id"] in captions:
            captions[row["image_id"]].append(target)
    features = np.stack([np.mean(captions[image_id], axis=0) for image_id in image_ids])
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def save_plot(figure, filename):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    path = DEBUG_DIR / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved {path}")


def plot_confusion_matrix(image_features, labels, prototypes):
    predictions = (image_features @ prototypes.T).argmax(axis=1)
    matrix = np.zeros((len(DEBUG_CLASSES), len(DEBUG_CLASSES)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)

    figure, axis = plt.subplots(figsize=(7, 6))
    heatmap = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(DEBUG_CLASSES)), DEBUG_CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(len(DEBUG_CLASSES)), DEBUG_CLASSES)
    axis.set_xlabel("Predicted group (nearest training-caption prototype)")
    axis.set_ylabel("Caption-keyword group")
    axis.set_title(f"{DEBUG_SPLIT}: approximate class confusion")
    for row in range(len(DEBUG_CLASSES)):
        for column in range(len(DEBUG_CLASSES)):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    figure.colorbar(heatmap, ax=axis, label="Images")
    save_plot(figure, "confusion_matrix.png")
    print(f"Group accuracy: {(predictions == labels).mean():.2%} (keyword labels, not retrieval Recall@1)")
    return matrix


def plot_clusters(image_features, caption_features, labels):
    # Fit one PCA projection to both modalities so their locations are comparable.
    combined = np.vstack([image_features, caption_features])
    combined -= combined.mean(axis=0)
    left, singular, _ = np.linalg.svd(combined, full_matrices=False)
    points = left[:, :2] * singular[:2]
    total_variance = (singular ** 2).sum()
    variance = singular[:2] ** 2 / total_variance if total_variance else np.zeros(2)
    count = len(labels)

    figure, axis = plt.subplots(figsize=(9, 7))
    colors = plt.get_cmap("tab10")
    for label, name in enumerate(DEBUG_CLASSES):
        selected = labels == label
        axis.scatter(*points[:count][selected].T, color=colors(label % 10), alpha=0.65, label=f"{name}: images")
        axis.scatter(*points[count:][selected].T, color=colors(label % 10), marker="x", label=f"{name}: captions")
    axis.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    axis.set_title("Image and mean-caption embeddings: shared 2D PCA")
    axis.legend()
    save_plot(figure, "class_clusters.png")


def plot_features(image_features, caption_features, labels):
    means = [np.stack([features[labels == i].mean(axis=0) for i in range(len(DEBUG_CLASSES))])
             for features in (image_features, caption_features)]
    limit = max(np.abs(mean[:, DEBUG_FEATURES]).max() for mean in means)
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(DEBUG_FEATURES)), 5))
    for axis, mean, title in zip(axes, means, ("Image features", "Mean-caption features")):
        heatmap = axis.imshow(mean[:, DEBUG_FEATURES], cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(DEBUG_FEATURES)), DEBUG_FEATURES)
        axis.set_yticks(range(len(DEBUG_CLASSES)), DEBUG_CLASSES)
        axis.set_xlabel("Embedding dimension")
        axis.set_title(title)
        figure.colorbar(heatmap, ax=axis, label="Mean value")
    figure.suptitle("Mean feature values by caption-keyword group")
    save_plot(figure, "embedding_features.png")


def plot_similarities(image_features, caption_features):
    matched = (image_features * caption_features).sum(axis=1)
    other = (image_features * np.roll(caption_features, 1, axis=0)).sum(axis=1)
    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 41)
    axis.hist(matched, bins=bins, alpha=0.6, label="Matching image captions")
    axis.hist(other, bins=bins, alpha=0.6, label="Other image captions")
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Images")
    axis.set_title("Can the CNN distinguish matching and other captions?")
    axis.legend()
    save_plot(figure, "caption_similarities.png")


def plot_training_history():
    if not HISTORY_PATH.exists():
        print("No training history yet; curves will be available after training.")
        return
    with open(HISTORY_PATH, newline="") as file:
        history = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["loss"]) for row in history])
    axes[0].set_ylabel("Training loss")
    axes[1].plot(epochs, [100 * float(row["recall_at_1"]) for row in history])
    axes[1].set_ylabel("Validation Recall@1 (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
    save_plot(figure, "training_curves.png")


def main():
    print(f"Groups: {', '.join(DEBUG_CLASSES)} (caption keywords; mixed groups excluded)")
    plot_training_history()
    prototypes = training_class_prototypes()
    dataset, features = load_image_gallery(DEBUG_SPLIT)
    indices, labels = select_samples(dataset)
    image_features = features.numpy()[indices]
    caption_features = average_captions(dataset, indices)
    plot_confusion_matrix(image_features, labels, prototypes)
    plot_clusters(image_features, caption_features, labels)
    plot_features(image_features, caption_features, labels)
    plot_similarities(image_features, caption_features)


if __name__ == "__main__":
    main()
