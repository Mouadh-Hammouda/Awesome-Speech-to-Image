"""Semantic map: images and their captions projected to 2-D in the shared embedding space, coloured by class."""

import csv
import signal
import sys
from collections import Counter
from pathlib import Path

# Allow direct execution from the repository root: python3 tools/semantic_map.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from common import CLASSES, DATA_DIR, model_file
from data_processing.caption_data import tokenise
from retrieval.caption_retrieval import BATCH_SIZE, device, embed_split, load_encoders

SPLIT = "val"
# The TRAIN_CLASSES the model to map was trained on; None is the model trained on every image.
MODEL_CLASSES = ("CAT", "DOG", "COMPUTER", "PEOPLE", "CAR")
# Names from common.CLASSES to show, e.g. ("CAT", "DOG"); None shows every class.
MAP_CLASSES = ("CAT", "DOG", "COMPUTER", "PEOPLE", "CAR")
# "tsne" separates clusters best; "pca" is instant and needs no scikit-learn.
METHOD = "tsne"
PERPLEXITY = 30
# An image belongs to a class if at least this many of its captions name it and no other class ties with it.
MIN_CAPTION_VOTES = 2
HOVER_PIXELS = 6
THUMBNAIL_SIZE = 200
TOOLTIP_OFFSET = 130
HOVER_CAPTIONS = 3


def read_captions(split):
    captions = {}
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        for row in csv.DictReader(file):
            captions.setdefault(row["image_id"], []).append(row["caption"])
    return captions


def image_class(captions):
    votes = Counter(name for caption in captions for name, class_words in CLASSES.items() if set(caption.split()) & class_words)
    best = votes.most_common(2)
    clear = best and best[0][1] >= MIN_CAPTION_VOTES and (len(best) == 1 or best[1][1] < best[0][1])
    return best[0][0] if clear else None


def centred(embeddings):
    # Images and captions are centred separately, so the map shows meaning, not the gap between the two modalities.
    return F.normalize(embeddings - embeddings.mean(dim=0), dim=-1)


def project(points):
    if METHOD == "pca":
        centred_points = points - points.mean(dim=0)
        _, _, components = torch.pca_lowrank(centred_points, q=2, center=False)
        return (centred_points @ components).numpy()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise SystemExit("t-SNE needs scikit-learn: pip install scikit-learn (or set METHOD = \"pca\")")
    return TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", random_state=0).fit_transform(points.numpy())


@torch.no_grad()
def embed_captions(text_encoder, vocabulary, captions):
    batches = tokenise(captions, vocabulary).split(BATCH_SIZE * 32)
    return torch.cat([text_encoder(batch.to(device)) for batch in batches]).cpu()


def main():
    class_names = list(MAP_CLASSES or CLASSES)
    unknown = [name for name in class_names if name not in CLASSES]
    if unknown:
        raise SystemExit(f"MAP_CLASSES {unknown} are not in common.CLASSES ({', '.join(CLASSES)})")

    image_encoder, text_encoder, vocabulary = load_encoders(MODEL_CLASSES)
    image_files, image_embeddings = embed_split(image_encoder, SPLIT, MODEL_CLASSES)
    captions_of = read_captions(SPLIT)

    # Only images with a clear class are mapped, together with all of their captions.
    image_classes = [image_class(captions_of.get(image_file.stem, [])) for image_file in image_files]
    kept = [image for image, name in enumerate(image_classes) if name in class_names]
    if not kept:
        raise SystemExit(f"No {SPLIT} images have a clear class among {', '.join(class_names)}")
    caption_texts, caption_images = [], []
    for image in kept:
        for caption in captions_of[image_files[image].stem]:
            caption_texts.append(caption)
            caption_images.append(image)

    image_points = centred(image_embeddings[kept].cpu())
    caption_points = centred(embed_captions(text_encoder, vocabulary, caption_texts))
    print(f"projecting {len(kept):,} images and {len(caption_texts):,} captions with {METHOD}...", flush=True)
    positions = project(torch.cat([image_points, caption_points]))
    image_positions, caption_positions = positions[:len(kept)], positions[len(kept):]
    image_colours = np.array([class_names.index(image_classes[image]) for image in kept])
    caption_colours = np.array([class_names.index(image_classes[image]) for image in caption_images])

    figure, axis = plt.subplots(figsize=(11, 9))
    colours = plt.get_cmap("tab20").colors
    for index, name in enumerate(class_names):
        colour = colours[index % len(colours)]
        axis.scatter(*caption_positions[caption_colours == index].T, marker="x", s=14, linewidths=0.8, alpha=0.5, color=colour)
        axis.scatter(*image_positions[image_colours == index].T, marker="o", s=30, edgecolors="white", linewidths=0.6, color=colour, label=name)
    axis_name = "t-SNE" if METHOD == "tsne" else "PC"
    axis.set(xlabel=f"{axis_name} 1", ylabel=f"{axis_name} 2")
    axis.set_title(f"{SPLIT} images (o) and their captions (x), coloured by class ({model_file(MODEL_CLASSES)})", fontsize=11)
    class_legend = axis.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(1, 1))
    axis.add_artist(class_legend)
    modalities = [Line2D([], [], marker="o", linestyle="", color="grey", label="image"),
                  Line2D([], [], marker="x", linestyle="", color="grey", label="caption")]
    axis.legend(handles=modalities, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(1, 0))
    figure.tight_layout()
    output_file = model_file(MODEL_CLASSES).parent / f"{SPLIT}_semantic_map.png"
    figure.savefig(output_file, dpi=150)
    print(f"wrote {output_file}")

    # Hovering an image shows its first captions; hovering a caption shows that caption and its image.
    hover_images = [image_files[image] for image in kept + caption_images]
    hover_texts = ["\n".join(caption.capitalize() for caption in captions_of[image_files[image].stem][:HOVER_CAPTIONS]) for image in kept]
    hover_texts += [caption.capitalize() for caption in caption_texts]
    show_on_hover(figure, axis, positions, hover_images, hover_texts)
    wake_timer = close_on_ctrl_c(figure)
    plt.show()
    wake_timer.stop()


def close_on_ctrl_c(figure):
    # The window's event loop hides Ctrl+C from Python; a periodic timer lets the handler run and close the window.
    signal.signal(signal.SIGINT, lambda *_: plt.close(figure))
    timer = figure.canvas.new_timer(interval=200)
    timer.add_callback(lambda: None)
    timer.start()
    return timer


def show_on_hover(figure, axis, positions, image_files, texts):
    thumbnail = OffsetImage(np.zeros((1, 1, 3)), zoom=1)
    tooltip = AnnotationBbox(thumbnail, (0, 0), xybox=(TOOLTIP_OFFSET, TOOLTIP_OFFSET), boxcoords="offset points")
    tooltip.set_visible(False)
    axis.add_artist(tooltip)
    caption = axis.text(0.0, -0.06, "", transform=axis.transAxes, fontsize=9, va="top")
    limits = (axis.get_xlim(), axis.get_ylim())

    def on_move(event):
        if event.inaxes is not axis:
            return
        distances = np.linalg.norm(axis.transData.transform(positions) - (event.x, event.y), axis=1)
        nearest = int(distances.argmin())
        visible = bool(distances[nearest] <= HOVER_PIXELS)
        tooltip.set_visible(visible)
        caption.set_text(texts[nearest] if visible else "")
        if visible:
            image = Image.open(image_files[nearest]).convert("RGB")
            image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            thumbnail.set_data(np.asarray(image))
            x, y = positions[nearest]
            tooltip.xy = (x, y)
            # Open towards the centre so the thumbnail stays on screen.
            right, top = x > np.mean(limits[0]), y > np.mean(limits[1])
            tooltip.xybox = (-TOOLTIP_OFFSET if right else TOOLTIP_OFFSET, -TOOLTIP_OFFSET if top else TOOLTIP_OFFSET)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_move)


if __name__ == "__main__":
    main()
