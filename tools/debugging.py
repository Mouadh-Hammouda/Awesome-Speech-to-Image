import csv
import math
import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 tools/debugging.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from data_processing.caption_data import class_votes, main_class
from models.caption_text_encoder import CaptionTextEncoder
from retrieval.caption_retrieval import device, embed_split, load_image_encoder
from common import CLASSES, DATA_DIR, caption_file

SPLIT = "val"
# The softmax temperature for turning image-to-class similarities into probabilities (these plots are only a diagnostic).
TEMPERATURE = 0.05
# Images without one clear class (see caption_data.main_class) are OTHER.
CLASS_NAMES = [*CLASSES, "OTHER"]
HOVER_PIXELS = 6


def read_captions(split):
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        return [(row["image_id"], row["caption"]) for row in csv.DictReader(file)]


def true_classes(image_files):
    captions = {}
    for image_id, caption in read_captions(SPLIT):
        captions.setdefault(image_id, []).append(set(caption.split()))
    classes = [main_class(captions[image_file.stem]) for image_file in image_files]
    return torch.tensor([len(CLASSES) if index is None else index for index in classes])


def class_embeddings(text_encoder):
    named = text_encoder.embed(list(CLASSES))
    # OTHER has no caption of its own, so it is the mean vector of the train captions that mention no class.
    other_rows = [row for row, (_, caption) in enumerate(read_captions("train")) if not any(class_votes([set(caption.split())]))]
    other = torch.from_numpy(np.load(caption_file("train"), mmap_mode="r")[other_rows])
    return torch.cat([named, F.normalize(other.mean(dim=0), dim=0).unsqueeze(0)])


@torch.no_grad()
def main():
    image_encoder = load_image_encoder()
    image_files, image_embeddings = embed_split(image_encoder, SPLIT)
    labels = true_classes(image_files)

    probabilities = (image_embeddings.to(device) @ class_embeddings(CaptionTextEncoder(device)).to(device).T / TEMPERATURE).softmax(dim=1).cpu()
    plot_class_circle(image_files, probabilities, labels)
    plot_confusion_matrix(probabilities, labels)
    plt.show()


def plot_class_circle(image_files, probabilities, labels):
    # Each image sits at the softmax-weighted average of the class anchors on a circle.
    angles = [math.pi / 2 - 2 * math.pi * index / len(CLASS_NAMES) for index in range(len(CLASS_NAMES))]
    anchors = torch.tensor([[math.cos(angle), math.sin(angle)] for angle in angles])
    positions = probabilities @ anchors

    figure, axis = plt.subplots(figsize=(8, 8))
    # Reversed so OTHER is drawn first, underneath the classes.
    for index, name in reversed(list(enumerate(CLASS_NAMES))):
        colour = "lightgrey" if name == "OTHER" else f"C{index}"
        axis.scatter(positions[labels == index, 0], positions[labels == index, 1], s=10, color=colour, label=name)
        axis.annotate(name, anchors[index] * 1.1, ha="center", va="center", fontweight="bold")
    axis.set_xlim(-1.2, 1.2)
    axis.set_ylim(-1.2, 1.2)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.legend()
    show_image_on_hover(figure, axis, image_files, positions)


def plot_confusion_matrix(probabilities, labels):
    # Rows are normalised so each shows how that true class is predicted, despite OTHER dominating the counts.
    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        probabilities.argmax(dim=1),
        labels=list(range(len(CLASS_NAMES))),
        display_labels=CLASS_NAMES,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
    )
    display.ax_.set_title(f"{SPLIT} confusion matrix")


def show_image_on_hover(figure, axis, image_files, positions):
    thumbnail = OffsetImage(np.zeros((1, 1, 3)), zoom=1)
    tooltip = AnnotationBbox(thumbnail, (0, 0), xybox=(120, 120), boxcoords="offset points")
    tooltip.set_visible(False)
    axis.add_artist(tooltip)

    def on_move(event):
        if event.inaxes is not axis:
            return
        distances = np.linalg.norm(axis.transData.transform(positions.numpy()) - (event.x, event.y), axis=1)
        nearest = distances.argmin()
        tooltip.set_visible(bool(distances[nearest] <= HOVER_PIXELS))
        if tooltip.get_visible():
            image = Image.open(image_files[nearest]).convert("RGB")
            image.thumbnail((200, 200))
            thumbnail.set_data(np.asarray(image))
            x, y = positions[nearest].tolist()
            tooltip.xy = (x, y)
            # Open towards the centre so the thumbnail stays on screen.
            tooltip.xybox = (-120 if x > 0 else 120, -120 if y > 0 else 120)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_move)


if __name__ == "__main__":
    main()
