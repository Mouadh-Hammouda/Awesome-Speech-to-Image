import argparse
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, CaptionEncoder, Split, hubness, load_model
from evaluate.metrics import caption_ranks, search_scores

SEARCH_SPLITS = ("val", "train")
TOP_K = 10
COLUMNS = 5
THUMBNAIL_SIZE = 220
DETAIL_SIZE = 640
OWN_IMAGE_COLOUR = "#1baf7a"
BLANK_COLOUR = "#e6e5e1"
CAPTION_COLOUR = "#52514e"


class SearchWindow:
    def __init__(self, root, image_model, encoder):
        self.root, self.image_model, self.encoder = root, image_model, encoder
        self.splits, self.embeddings, self.hubness = {}, {}, {}
        self.photos = []
        self.own_image = None
        root.title(f"Caption search: {image_model.dir} (epoch {image_model.epoch})")
        self.build_search_bar()
        self.build_options()
        self.status = tk.StringVar(value="Type a caption and press Enter, or click Random caption.")
        ttk.Label(root, textvariable=self.status, padding=(10, 0, 10, 6), font=("TkDefaultFont", 11)).pack(fill="x")
        self.build_tiles()
        self.prepare()

    def build_search_bar(self):
        bar = ttk.Frame(self.root, padding=10)
        bar.pack(fill="x")
        self.query = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.query, font=("TkDefaultFont", 13))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _: self.search())
        entry.focus_set()
        ttk.Button(bar, text="Search", command=self.search).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Random caption", command=self.random_caption).pack(side="left", padx=(8, 0))
        ttk.Label(bar, text="Images:").pack(side="left", padx=(16, 4))
        self.split_name = tk.StringVar(value=SEARCH_SPLITS[0])
        choice = ttk.Combobox(bar, textvariable=self.split_name, values=SEARCH_SPLITS, state="readonly", width=6)
        choice.pack(side="left")
        choice.bind("<<ComboboxSelected>>", lambda _: self.search(self.own_image))

    def build_options(self):
        options = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        options.pack(fill="x")
        self.crops, self.use_hubness = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        for text, variable in (("Extra crops", self.crops), ("Hubness correction", self.use_hubness)):
            ttk.Checkbutton(options, text=text, variable=variable, command=lambda: self.search(self.own_image)).pack(
                side="left", padx=(0, 12))

    def build_tiles(self):
        grid = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        grid.pack()
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), BLANK_COLOUR))
        self.tiles = []
        for index in range(TOP_K):
            tile = tk.Frame(grid, highlightthickness=4, highlightbackground=self.root.cget("bg"))
            tile.grid(row=index // COLUMNS, column=index % COLUMNS, padx=4, pady=4, sticky="n")
            picture = tk.Label(tile, image=self.blank, width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE, cursor="hand2")
            picture.pack()
            score = ttk.Label(tile, font=("TkDefaultFont", 10, "bold"))
            score.pack(anchor="w")
            caption = ttk.Label(tile, wraplength=THUMBNAIL_SIZE, font=("TkDefaultFont", 9), foreground=CAPTION_COLOUR)
            caption.pack(anchor="w")
            self.tiles.append((tile, picture, score, caption))

    def wait(self, message):
        self.status.set(message)
        self.root.update_idletasks()

    def prepare(self):
        name, crops = self.split_name.get(), self.crops.get()
        if name not in self.splits:
            self.splits[name] = Split(name)
        data = self.splits[name]
        key = (name, crops)
        if key not in self.embeddings:
            self.wait(f"Embedding the {len(data.image_ids):,} {name} images, progress in the terminal...")
            self.embeddings[key] = self.image_model.embed(data, crops=crops)
        if not self.use_hubness.get():
            return data, self.embeddings[key], None
        if key not in self.hubness:
            self.wait("Measuring each image's hubness against the train captions...")
            self.hubness[key] = hubness(self.embeddings[key])
        return data, self.embeddings[key], self.hubness[key]

    def random_caption(self):
        data = self.prepare()[0]
        caption = random.randrange(len(data.captions))
        self.query.set(data.captions[caption])
        self.search(own_image=data.caption_images[caption].item())

    def search(self, own_image=None):
        caption = self.query.get().strip()
        if not caption:
            return
        data, images, image_hubness = self.prepare()
        if own_image is None:
            matches = [number for number, text in enumerate(data.captions) if text.lower() == caption.lower()]
            own_image = data.caption_images[matches[0]].item() if matches else None
        self.own_image = own_image
        query = self.encoder([caption])
        top_images = search_scores(query, images, image_hubness)[0].topk(TOP_K).indices
        similarities = (query @ images.T)[0, top_images]
        self.show(data, top_images.tolist(), similarities.tolist(), own_image)

        status = f"Top {TOP_K} of {len(data.image_ids):,} {self.split_name.get()} images."
        if own_image is not None:
            rank = caption_ranks(query, images, data.caption_images.new_tensor([own_image]), image_hubness).item() + 1
            status += f"  Own image: rank {rank:,}" + (" (outlined in green)." if rank <= TOP_K else ", not shown.")
        self.status.set(status)

    def show(self, data, images, similarities, own_image):
        self.photos.clear()
        for rank, (tile, image, similarity) in enumerate(zip(self.tiles, images, similarities), start=1):
            frame, picture, score, caption = tile
            thumbnail = Image.open(data.image_file(image)).convert("RGB")
            thumbnail.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            photo = ImageTk.PhotoImage(thumbnail)
            self.photos.append(photo)
            picture.configure(image=photo)
            picture.bind("<Button-1>", lambda _, image=image: self.show_details(data, image))
            score.configure(text=f"#{rank}   similarity {similarity:.3f}")
            caption.configure(text=data.image_captions[image][0])
            colour = OWN_IMAGE_COLOUR if image == own_image else self.root.cget("bg")
            frame.configure(highlightbackground=colour, highlightcolor=colour)

    def show_details(self, data, image):
        window = tk.Toplevel(self.root)
        window.title(data.image_ids[image])
        picture = Image.open(data.image_file(image)).convert("RGB")
        picture.thumbnail((DETAIL_SIZE, DETAIL_SIZE))
        photo = ImageTk.PhotoImage(picture)
        label = ttk.Label(window, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        captions = "\n".join(f"• {caption}" for caption in data.image_captions[image])
        ttk.Label(window, text=f"Image {data.image_ids[image]} ({data.base})\n{captions}", justify="left",
                  wraplength=DETAIL_SIZE, padding=(10, 0, 10, 10)).pack(anchor="w")


def main():
    parser = argparse.ArgumentParser(description="Search images by caption.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help=MODEL_HELP)
    image_model = load_model(parser.parse_args().model)
    print(image_model.label, flush=True)
    encoder = CaptionEncoder()
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    SearchWindow(root, image_model, encoder)
    root.mainloop()


if __name__ == "__main__":
    main()
