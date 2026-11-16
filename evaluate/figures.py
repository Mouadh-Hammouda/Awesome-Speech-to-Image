import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.axisbelow": True, "font.size": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def thumbnail(axis, data, image, title, colour=None, size=256):
    picture = Image.open(data.image_file(image)).convert("RGB")
    picture.thumbnail((size, size))
    square = Image.new("RGB", (size, size), "white")
    square.paste(picture, ((size - picture.width) // 2, (size - picture.height) // 2))
    axis.imshow(square)
    axis.set_title(title, fontsize=9, loc="center", fontweight="normal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(colour is not None)
        spine.set_color(colour or "none")
        spine.set_linewidth(4)


def save(figure, path, dpi=150):
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    print(f"wrote {path}")
