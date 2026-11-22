"""Training curves of one or more runs, from their training_metrics.csv, and the class confusion matrix plot."""

import csv
import sys
from pathlib import Path

# Allow direct execution from the repository root: python3 tools/plot_training.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Figure is used without pyplot, so plotting needs no window and is safe inside the training script.
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from common import RESULTS_DIR

# Run folders to plot, e.g. ("results/image_to_caption",); empty plots every folder under results/ with a training_metrics.csv.
RUNS = ()
# Each panel: title, then (CSV column, label, line style) for every line in it. Columns a run's CSV lacks are skipped.
PANELS = [
    ("loss", [("val loss", "val", "-"), ("train eval loss", "train (2,000 images)", "--"), ("loss", "train, during training (augmented)", ":"), ("one-hot loss", "one-hot, during training", "-.")]),
    ("caption -> image R@1 (%)", [("val t2i R@1", "val", "-"), ("train t2i R@1", "train (2,000 images)", "--")]),
    ("caption -> image R@k (%)", [("val t2i R@1", "val R@1", "-"), ("val t2i R@5", "val R@5", "--"), ("val t2i R@10", "val R@10", ":"), ("train t2i R@10", "train R@10", "-.")]),
    ("image -> caption R@k (%), val", [("val i2t R@1", "R@1", "-"), ("val i2t R@5", "R@5", "--"), ("val i2t R@10", "R@10", ":")]),
    ("caption -> image MRR, val", [("val t2i MRR", "MRR", "-")]),
    ("caption -> image median rank, val", [("val t2i median rank", "median rank", "-")]),
    ("class P@10 (%), val, average over classes", [("val class P@10", "P@10", "-")]),
    ("class lift@10 (x chance), val", [("val class lift@10", "lift@10", "-")]),
    ("cosine of caption-image pairs, val", [("val t2i pos cos", "matching", "-"), ("val t2i neg cos", "non-matching", "--")]),
    ("image effective rank, val", [("val image effective rank", "effective rank", "-")]),
    ("learned temperature", [("temperature", "temperature", "-")]),
    ("learning rate", [("lr", "learning rate", "-")]),
]


def read_metrics(folder):
    with open(Path(folder) / "training_metrics.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    return {key: [float(row[key]) for row in rows] for key in rows[0]} if rows else {}


def plot_runs(folders, output_file):
    figure = Figure(figsize=(20, 13))
    axes = figure.subplots(3, 4).flatten()
    for run_index, folder in enumerate(folders):
        metrics = read_metrics(folder)
        if not metrics:
            continue
        for axis, (_, lines) in zip(axes, PANELS):
            for line_index, (key, label, style) in enumerate(lines):
                # The one-hot loss is only worth a line when the training loss differs from it (soft targets).
                if key not in metrics or (key == "one-hot loss" and metrics[key] == metrics["loss"]):
                    continue
                # One run: a colour per line. Several runs: a colour per run (figure legend) and the line style tells the curves apart.
                colour = f"C{run_index}" if len(folders) > 1 else f"C{line_index}"
                seen = label in axis.get_legend_handles_labels()[1]
                axis.plot(metrics["epoch"], metrics[key], style, color=colour, linewidth=1.4, label=None if seen else label)
        # The epoch saved as the best model.
        best = max(range(len(metrics["epoch"])), key=lambda row: metrics["val t2i R@1"][row])
        colour = f"C{run_index}" if len(folders) > 1 else "C0"
        axes[1].plot(metrics["epoch"][best], metrics["val t2i R@1"][best], "o", color=colour)
        axes[1].annotate(f"{metrics['val t2i R@1'][best]:.1f} (epoch {metrics['epoch'][best]:.0f})", (metrics["epoch"][best], metrics["val t2i R@1"][best]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=colour)
    for axis, (title, _) in zip(axes, PANELS):
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if len(axis.get_legend_handles_labels()[0]) > 1 or len(folders) > 1:
            axis.legend(fontsize=7, frameon=False)
    if len(folders) > 1:
        runs = [Line2D([], [], color=f"C{index}", linewidth=2, label=Path(folder).name) for index, folder in enumerate(folders)]
        figure.legend(handles=runs, loc="upper center", ncol=len(runs), fontsize=10, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.suptitle(str(folders[0]), fontsize=11, color="grey")
        figure.tight_layout()
    figure.savefig(output_file, dpi=110)


def plot_confusion(matrix, class_names, output_file, title):
    # matrix from training.metrics.class_confusion: rows are the class of the caption's image, columns the classes of its top-10 images (%).
    figure = Figure(figsize=(8.5, 7.5))
    axis = figure.subplots()
    image = axis.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=100)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column].item()
            axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=9, color="white" if value > 50 else "black")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("class of the retrieved top-10 images")
    axis.set_ylabel("class of the caption's image")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, label="% of the top-10 images", shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_file, dpi=120)


def main():
    folders = [Path(run) for run in RUNS] or sorted(folder for folder in RESULTS_DIR.parent.iterdir() if (folder / "training_metrics.csv").exists())
    if not folders:
        raise SystemExit(f"No training_metrics.csv found under {RESULTS_DIR.parent}")
    for folder in folders:
        plot_runs([folder], folder / "training_curves.png")
        print(f"wrote {folder / 'training_curves.png'}")
    if len(folders) > 1:
        output_file = RESULTS_DIR.parent / "training_comparison.png"
        plot_runs(folders, output_file)
        print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
