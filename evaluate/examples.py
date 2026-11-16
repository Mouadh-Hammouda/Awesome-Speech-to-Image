import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, Split, caption_embeddings, evaluation_dir, load_model
from evaluate.figures import AQUA, save, thumbnail
from evaluate.metrics import caption_ranks

EXAMPLES = 6
EXAMPLE_TOP = 4
HUBS = 10
COLUMNS = 5
CAPTION_LENGTH = 34


def shorten(text):
    return text if len(text) <= CAPTION_LENGTH else text[:CAPTION_LENGTH - 2].rstrip() + "…"


def plot_examples(path, title, data, captions, images, ranks, chosen):
    figure = plt.figure(figsize=(2.6 * (EXAMPLE_TOP + 1), 3.1 * len(chosen) + 0.6), layout="constrained")
    figure.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    for row, caption in zip(figure.subfigures(len(chosen), 1), chosen):
        row.suptitle(f"“{data.captions[caption]}”", x=0.01, ha="left", fontsize=11, fontweight="bold")
        axes = row.subplots(1, EXAMPLE_TOP + 1)
        own = data.caption_images[caption].item()
        top = (captions[caption] @ images.T).topk(EXAMPLE_TOP).indices.tolist()
        thumbnail(axes[0], data, own, f"own image, rank {ranks[caption].item() + 1}", colour=AQUA)
        for rank, (axis, image) in enumerate(zip(axes[1:], top), start=1):
            thumbnail(axis, data, image, f"CNN's #{rank}", colour=AQUA if image == own else None)
    save(figure, path, dpi=120)


def plot_hubs(path, data, captions, images):
    first = torch.cat([(chunk @ images.T).argmax(1) for chunk in captions.split(4096)]).cpu()
    counts = torch.bincount(first, minlength=len(images))
    rows = math.ceil(HUBS / COLUMNS)
    figure, axes = plt.subplots(rows, COLUMNS, figsize=(14, 3.6 * rows + 0.5), layout="constrained")
    for axis, image in zip(axes.flat, counts.argsort(descending=True)[:HUBS].tolist()):
        title = f"first for {counts[image].item()} captions\n“{shorten(data.image_captions[image][0])}”"
        thumbnail(axis, data, image, title)
    figure.suptitle(f"Hub images: first for far more captions than the ~{len(captions) / len(images):.0f} expected. "
                    f"{(counts == 0).sum().item()} of {len(images):,} images are never first.",
                    x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(figure, path, dpi=120)


def plot_all_examples(image_model, seed=0):
    folder = evaluation_dir(image_model)
    val = Split("val")
    images, captions = image_model.embed(val), caption_embeddings(val)
    ranks = caption_ranks(captions, images, val.caption_images)
    generator = np.random.default_rng(seed)
    hits = torch.nonzero(ranks == 0).squeeze(1).numpy()
    misses = torch.nonzero(ranks >= 10).squeeze(1).numpy()
    plot_examples(folder / "successes.png", "Own image first", val, captions, images, ranks,
                  generator.choice(hits, EXAMPLES, replace=False).tolist())
    plot_examples(folder / "failures.png", "Own image not in the top 10", val, captions, images, ranks,
                  generator.choice(misses, EXAMPLES, replace=False).tolist())
    plot_hubs(folder / "hubs.png", val, captions, images)


def main():
    parser = argparse.ArgumentParser(description="Example successes, failures and hubs of a trained image CNN on val.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help=MODEL_HELP)
    parser.add_argument("--seed", type=int, default=0, help="which random examples to show")
    arguments = parser.parse_args()
    image_model = load_model(arguments.model)
    print(image_model.label, flush=True)
    plot_all_examples(image_model, arguments.seed)


if __name__ == "__main__":
    main()
