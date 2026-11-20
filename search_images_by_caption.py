import re
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox
from PIL import Image
from caption_data import centre_crop, load_images, tokenise
from caption_models import CaptionTextEncoder, ImageEncoderCNN, model_file
from utils import RAW_DIR, get_device

SEARCH_SPLITS = ("val", "train", "test")
# The TRAIN_CLASSES the model to search with was trained on; None is the model trained on every image.
MODEL_CLASSES = None
BATCH_SIZE = 128
TOP_K = 10
GRID_ROWS, GRID_COLUMNS = 2, 5

device = get_device()


def load_encoders(class_names=None):
    checkpoint = torch.load(model_file(class_names), map_location=device)
    vocabulary = checkpoint["vocabulary"]
    image_encoder = ImageEncoderCNN().to(device)
    text_encoder = CaptionTextEncoder(len(vocabulary) + 2).to(device)
    image_encoder.load_state_dict(checkpoint["image_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    return image_encoder.eval(), text_encoder.eval(), vocabulary


def clean_caption(caption):
    # The training captions are uppercase letters and digits only.
    return re.sub(r"[^A-Z0-9 ]", "", caption.upper())


@torch.no_grad()
def embed_split(image_encoder, split, class_names=None):
    trained_file = model_file(class_names)
    embeddings_file = trained_file.parent / f"{split}_image_embeddings.pt"
    # Saved embeddings are out of date once the model has been retrained.
    if embeddings_file.exists() and embeddings_file.stat().st_mtime > trained_file.stat().st_mtime:
        saved = torch.load(embeddings_file, map_location=device)
        image_ids, image_embeddings = saved["image_ids"], saved["image_embeddings"]
    else:
        image_ids = sorted(image_file.stem for image_file in (RAW_DIR / split).glob("*.jpg"))
        images = load_images(split, image_ids)
        image_embeddings = torch.cat([image_encoder(centre_crop(batch).to(device)) for batch in images.split(BATCH_SIZE)])
        torch.save({"image_ids": image_ids, "image_embeddings": image_embeddings.cpu()}, embeddings_file)
    return [RAW_DIR / split / f"{image_id}.jpg" for image_id in image_ids], image_embeddings


@torch.no_grad()
def find_matches(caption, text_encoder, vocabulary, image_embeddings):
    scores = text_encoder(tokenise([caption], vocabulary).to(device)) @ image_embeddings.T
    top_scores, top_images = scores[0].topk(min(TOP_K, len(image_embeddings)))
    return top_images.tolist(), top_scores.tolist()


def show_matches(axes, image_files, scores):
    for axis in axes.flat:
        axis.clear()
        axis.axis("off")
    for rank, (axis, image_file, score) in enumerate(zip(axes.flat, image_files, scores), start=1):
        axis.imshow(Image.open(image_file).convert("RGB"))
        axis.set_title(f"#{rank} ({score:.3f})")


def main():
    print("Loading model...", flush=True)
    image_encoder, text_encoder, vocabulary = load_encoders(MODEL_CLASSES)
    # Fetched the first time each split is selected, so startup only pays for the first one.
    embedded_splits = {SEARCH_SPLITS[0]: embed_split(image_encoder, SEARCH_SPLITS[0], MODEL_CLASSES)}

    figure, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=(15, 8))
    figure.canvas.manager.set_window_title("Caption search")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.15, wspace=0.05, hspace=0.15)
    figure.suptitle("Type a caption below and press Enter")
    for axis in axes.flat:
        axis.axis("off")
    split_buttons = RadioButtons(figure.add_axes([0.01, 0.01, 0.07, 0.11]), SEARCH_SPLITS)
    text_box = TextBox(figure.add_axes([0.14, 0.04, 0.84, 0.05]), "Caption ")

    def search(typed_caption):
        caption = clean_caption(typed_caption)
        words = caption.split()
        if not words:
            return
        unknown_words = [word for word in words if word not in vocabulary]
        if len(unknown_words) == len(words):
            figure.suptitle("None of those words are in the vocabulary, try different ones")
        else:
            image_files, image_embeddings = embedded_splits[split_buttons.value_selected]
            top_images, top_scores = find_matches(caption, text_encoder, vocabulary, image_embeddings)
            show_matches(axes, [image_files[image] for image in top_images], top_scores)
            figure.suptitle(f"{split_buttons.value_selected}: {caption}" + (f"\nNot in the vocabulary: {', '.join(unknown_words)}" if unknown_words else ""))
        figure.canvas.draw_idle()

    def change_split(split):
        if split not in embedded_splits:
            figure.suptitle(f"Loading {split} images...")
            figure.canvas.draw()
            figure.canvas.flush_events()
            embedded_splits[split] = embed_split(image_encoder, split, MODEL_CLASSES)
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
