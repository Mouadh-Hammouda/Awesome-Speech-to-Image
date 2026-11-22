"""Semantic map: images and their captions projected to 2-D, side by side, coloured by class."""

import csv
import signal
import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 tools/semantic_map.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from common import CLASSES, DATA_DIR, caption_file, model_file
from data_processing.caption_data import MIN_CAPTION_VOTES, main_class
from retrieval.caption_retrieval import embed_split, load_image_encoder

SPLIT = "val"
# Names from common.CLASSES to show, e.g. ("CAT", "DOG"); None shows every class.
MAP_CLASSES = ("CAT", "DOG", "COMPUTER", "PEOPLE", "CAR")
# Also draw the split's other images (no clear class, or a class not shown) as grey dots, so the classes are seen against everything else.
SHOW_OTHER_IMAGES = True
# "tsne" separates clusters best; "pca" is instant and needs no scikit-learn.
METHOD = "tsne"
PERPLEXITY = 30
HOVER_PIXELS = 6
THUMBNAIL_SIZE = 200
TOOLTIP_OFFSET = 130
HOVER_CAPTIONS = 3
OTHER_COLOUR = "#c8c8c8"


def read_captions(split):
    # Each image's captions and, for each, its row in the metadata (which is its row in the caption vectors).
    captions, rows = {}, {}
    with open(DATA_DIR / f"{split}_meta.csv", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file)):
            captions.setdefault(row["image_id"], []).append(row["caption"])
            rows.setdefault(row["image_id"], []).append(row_number)
    return captions, rows


def image_class(captions):
    # The class name of the image if it has one clear class (the same rule as the training metrics), else None.
    index = main_class([set(caption.split()) for caption in captions])
    return None if index is None else list(CLASSES)[index]


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


def main():
    class_names = list(MAP_CLASSES or CLASSES)
    unknown = [name for name in class_names if name not in CLASSES]
    if unknown:
        raise SystemExit(f"MAP_CLASSES {unknown} are not in common.CLASSES ({', '.join(CLASSES)})")

    image_encoder = load_image_encoder()
    image_files, image_embeddings = embed_split(image_encoder, SPLIT)
    captions_of, rows_of = read_captions(SPLIT)

    # Images with a clear class are mapped together with all of their captions; the rest only as grey context.
    image_classes = [image_class(captions_of.get(image_file.stem, [])) for image_file in image_files]
    kept = [image for image, name in enumerate(image_classes) if name in class_names]
    if not kept:
        raise SystemExit(f"No {SPLIT} images have a clear class among {', '.join(class_names)}")
    others = [image for image, name in enumerate(image_classes) if name not in class_names] if SHOW_OTHER_IMAGES else []
    caption_texts, caption_images, caption_rows = [], [], []
    for image in kept:
        for caption, row in zip(captions_of[image_files[image].stem], rows_of[image_files[image].stem]):
            caption_texts.append(caption)
            caption_images.append(image)
            caption_rows.append(row)

    mapped_images = kept + others
    image_points = centred(image_embeddings[mapped_images].cpu())
    # The caption vectors were computed once, so no text model is needed here.
    caption_points = centred(torch.from_numpy(np.load(caption_file(SPLIT), mmap_mode="r")[caption_rows]))
    # Each space gets its own projection: a caption and its image are only about 0.25 apart in cosine, so one joint
    # projection would put all images on one side and all captions on the other. Matching colours show the alignment.
    print(f"projecting {len(mapped_images):,} images and {len(caption_texts):,} captions with {METHOD}...", flush=True)
    image_positions, caption_positions = project(image_points), project(caption_points)
    image_colours = np.array([class_names.index(image_classes[image]) for image in kept])
    caption_colours = np.array([class_names.index(image_classes[image]) for image in caption_images])

    figure, (image_axis, caption_axis) = plt.subplots(1, 2, figsize=(17, 8.5))
    colours = plt.get_cmap("tab10").colors
    if others:
        image_axis.scatter(*image_positions[len(kept):].T, marker="o", s=8, linewidths=0, color=OTHER_COLOUR, zorder=1)
    draw_classes(image_axis, image_positions[:len(kept)], image_colours, class_names, colours, marker="o", size=30)
    draw_classes(caption_axis, caption_positions, caption_colours, class_names, colours, marker="x", size=14)
    image_axis.set_title(f"{SPLIT} images, embedded by the CNN", fontsize=12)
    caption_axis.set_title(f"their captions, embedded by the text model (the CNN's target)", fontsize=12)
    handles = [Line2D([], [], marker="o", linestyle="", color=colours[index % len(colours)], label=f"{name} ({int((image_colours == index).sum())} images)") for index, name in enumerate(class_names)]
    if others:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=4, color=OTHER_COLOUR, label=f"other images ({len(others)})"))
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    figure.suptitle(f"{METHOD} map; an image's class is the one at least {MIN_CAPTION_VOTES} of its captions name ({model_file()})", fontsize=9, color="grey")
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_file = model_file().parent / f"{SPLIT}_semantic_map.png"
    figure.savefig(output_file, dpi=150)
    print(f"wrote {output_file}")

    # Hovering an image shows it with its first captions; hovering a caption shows that caption and its image.
    image_texts = ["\n".join(caption.capitalize() for caption in captions_of[image_files[image].stem][:HOVER_CAPTIONS]) for image in mapped_images]
    show_on_hover(figure, image_axis, image_positions, [image_files[image] for image in mapped_images], image_texts)
    show_on_hover(figure, caption_axis, caption_positions, [image_files[image] for image in caption_images], [caption.capitalize() for caption in caption_texts])
    wake_timer = close_on_ctrl_c(figure)
    plt.show()
    wake_timer.stop()


def draw_classes(axis, positions, colours_of, class_names, colours, marker, size):
    for index, name in enumerate(class_names):
        colour = colours[index % len(colours)]
        points = positions[colours_of == index]
        if marker == "x":
            axis.scatter(*points.T, marker=marker, s=size, linewidths=0.8, alpha=0.6, color=colour, zorder=2)
        else:
            axis.scatter(*points.T, marker=marker, s=size, edgecolors="white", linewidths=0.5, color=colour, zorder=2)
        # The class name at the middle of its points, so the map reads without the legend.
        label = axis.text(*np.median(points, axis=0), name, color=colour, fontsize=11, fontweight="bold", ha="center", va="center", zorder=3)
        label.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(False)


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
