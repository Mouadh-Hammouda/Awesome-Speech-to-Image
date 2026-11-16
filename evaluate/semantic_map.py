import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, Split, caption_embeddings, evaluation_dir, load_model
from evaluate.concepts import NAMES, NO_CONCEPT, concept_labels

METHODS = ("tsne", "pca")
PERPLEXITY = 30
HOVER_PIXELS = 6
THUMBNAIL_SIZE = 200
TOOLTIP_OFFSET = 130


def project(points, method):
    points = points.cpu()
    if method == "pca":
        centred = points - points.mean(dim=0)
        _, _, components = torch.pca_lowrank(centred, q=2, center=False)
        return (centred @ components).numpy()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise SystemExit("t-SNE needs scikit-learn: pip install scikit-learn (or use --method pca)")
    return TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", random_state=0).fit_transform(points.numpy())


def centred(embeddings):
    # Images and captions are centred separately, so the map shows meaning, not the gap between the two modalities.
    return F.normalize(embeddings - embeddings.mean(dim=0), dim=-1)


def add_hover(figure, axis, data, positions, images, texts):
    picture = OffsetImage(np.zeros((1, 1, 3)), zoom=1)
    tooltip = AnnotationBbox(picture, (0, 0), xybox=(TOOLTIP_OFFSET, TOOLTIP_OFFSET), boxcoords="offset points")
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
            thumbnail = Image.open(data.image_file(images[nearest])).convert("RGB")
            thumbnail.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            picture.set_data(np.asarray(thumbnail))
            x, y = positions[nearest]
            tooltip.xy = (x, y)
            right, top = x > np.mean(limits[0]), y > np.mean(limits[1])
            tooltip.xybox = (-TOOLTIP_OFFSET if right else TOOLTIP_OFFSET, -TOOLTIP_OFFSET if top else TOOLTIP_OFFSET)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_move)


def draw_map(image_model, chosen=NAMES, method="tsne", show=True):
    val = Split("val")
    labels = concept_labels(val).tolist()
    kept = [image for image, label in enumerate(labels) if label != NO_CONCEPT and NAMES[label] in chosen]
    kept_set = set(kept)
    caption_rows = [row for row, image in enumerate(val.caption_images.tolist()) if image in kept_set]
    caption_images = val.caption_images[caption_rows].tolist()

    image_points = centred(image_model.embed(val)[kept])
    caption_points = centred(caption_embeddings(val)[caption_rows])
    positions = project(torch.cat([image_points, caption_points]), method)
    image_positions, caption_positions = positions[:len(kept)], positions[len(kept):]
    image_concepts = np.array([chosen.index(NAMES[labels[image]]) for image in kept])
    caption_concepts = np.array([chosen.index(NAMES[labels[image]]) for image in caption_images])

    figure, axis = plt.subplots(figsize=(11, 9))
    colours = plt.get_cmap("tab20").colors
    for index, name in enumerate(chosen):
        colour = colours[index % len(colours)]
        axis.scatter(*caption_positions[caption_concepts == index].T, marker="x", s=14, linewidths=0.8, alpha=0.5,
                     color=colour)
        axis.scatter(*image_positions[image_concepts == index].T, marker="o", s=30, edgecolors="white", linewidths=0.6,
                     color=colour, label=name)
    axis_name = "t-SNE" if method == "tsne" else "PC"
    axis.set(xlabel=f"{axis_name} 1", ylabel=f"{axis_name} 2")
    axis.set_title(f"Val images (o) and their captions (x), coloured by concept ({image_model.dir}, "
                   f"epoch {image_model.epoch})", fontsize=11)
    markers = [Line2D([], [], marker="o", linestyle="", color="grey", label="image"),
               Line2D([], [], marker="x", linestyle="", color="grey", label="caption")]
    concept_legend = axis.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(1, 1))
    axis.add_artist(concept_legend)
    axis.legend(handles=markers, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(1, 0))
    figure.tight_layout()
    path = evaluation_dir(image_model) / "semantic_map.png"
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")
    if show:
        texts = ["\n".join(val.image_captions[image][:3]) for image in kept] + [val.captions[row] for row in caption_rows]
        add_hover(figure, axis, val, positions, kept + caption_images, texts)
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Semantic map of a trained image CNN on val (hover = image).")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help=MODEL_HELP)
    parser.add_argument("--concepts", nargs="+", default=NAMES, choices=NAMES, metavar="CONCEPT",
                        help=f"concepts to show (default all: {', '.join(NAMES)})")
    parser.add_argument("--method", default="tsne", choices=METHODS, help="how the embeddings are projected to x, y")
    arguments = parser.parse_args()
    image_model = load_model(arguments.model)
    print(image_model.label, flush=True)
    draw_map(image_model, arguments.concepts, arguments.method)


if __name__ == "__main__":
    main()
