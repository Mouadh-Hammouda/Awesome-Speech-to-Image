import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 retrieval/search_images_by_caption.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox
from PIL import Image
from models.caption_text_encoder import CaptionTextEncoder
from common import FEATURES_DIR, RAW_DIR
from retrieval.caption_retrieval import device, load_image_encoder
from retrieval.improved_search import search_index, search_scores

SEARCH_SPLITS = ("val", "train")
TOP_K = 10
GRID_ROWS, GRID_COLUMNS = 2, 5


@torch.no_grad()
def find_matches(caption, text_encoder, image_embeddings, image_hubness):
    scores = search_scores(text_encoder.embed([caption]).to(device), image_embeddings, image_hubness)
    top_scores, top_images = scores[0].topk(min(TOP_K, len(image_embeddings)))
    return top_images.tolist(), top_scores.tolist()


def load_split(image_encoder, split):
    image_ids = (FEATURES_DIR / f"{split}_image_ids.txt").read_text().split()
    return [RAW_DIR / split / f"{image_id}.jpg" for image_id in image_ids], *search_index(image_encoder, split, device)


def show_matches(axes, image_files, scores):
    for axis in axes.flat:
        axis.clear()
        axis.axis("off")
    for rank, (axis, image_file, score) in enumerate(zip(axes.flat, image_files, scores), start=1):
        axis.imshow(Image.open(image_file).convert("RGB"))
        axis.set_title(f"#{rank} ({score:.3f})")


def main():
    print("Loading model...", flush=True)
    image_encoder = load_image_encoder()
    text_encoder = CaptionTextEncoder(device)
    # Fetched the first time each split is selected, so startup only pays for the first one.
    embedded_splits = {SEARCH_SPLITS[0]: load_split(image_encoder, SEARCH_SPLITS[0])}

    figure, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=(15, 8))
    figure.canvas.manager.set_window_title("Caption search")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.15, wspace=0.05, hspace=0.15)
    figure.suptitle("Type a caption below and press Enter")
    for axis in axes.flat:
        axis.axis("off")
    split_buttons = RadioButtons(figure.add_axes([0.01, 0.01, 0.07, 0.11]), SEARCH_SPLITS)
    text_box = TextBox(figure.add_axes([0.14, 0.04, 0.84, 0.05]), "Caption ")

    def search(typed_caption):
        caption = typed_caption.strip()
        if not caption:
            return
        image_files, image_embeddings, image_hubness = embedded_splits[split_buttons.value_selected]
        top_images, top_scores = find_matches(caption, text_encoder, image_embeddings, image_hubness)
        show_matches(axes, [image_files[image] for image in top_images], top_scores)
        figure.suptitle(f"{split_buttons.value_selected}: {caption}")
        figure.canvas.draw_idle()

    def change_split(split):
        if split not in embedded_splits:
            figure.suptitle(f"Loading {split} images...")
            figure.canvas.draw()
            figure.canvas.flush_events()
            embedded_splits[split] = load_split(image_encoder, split)
            figure.suptitle(f"{split} ready, type a caption below and press Enter")
            figure.canvas.draw_idle()
        search(text_box.text)
        text_box.begin_typing()

    text_box.on_submit(search)
    split_buttons.on_clicked(change_split)
    text_box.begin_typing()
    plt.show()


if __name__ == "__main__":
    main()
