import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, Split, caption_embeddings, evaluation_dir, hubness, load_model
from evaluate.figures import AQUA, BLUE, GREY, save
from evaluate.metrics import RECALL_KS, caption_ranks, image_ranks, recall

COMPARISON_FILE = Path("results/model_comparison.csv")
COMMON_WORDS = 30
SEED = 0
STOPWORDS = set("""A AN THE OF ON IN AT TO WITH AND OR IS ARE IT ITS THIS THAT THERE SOME TWO THREE ONE FOR FROM BY
NEXT UP DOWN OUT OVER NEAR WHILE HIS HER THEIR THEY HE SHE WHO BEING HAS HAVE AS INTO OTHER EACH VERY LOOKING""".split())


def direction_metrics(ranks):
    return {**{name: round(value, 2) for name, value in recall(ranks).items()}, "median_rank": int(ranks.median()) + 1}


def format_recall(metrics):
    return "/".join(f"{metrics[f'R@{k}']:.1f}" for k in RECALL_KS)


def plot_ranks(path, ranks, captions, images, caption_images):
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    positions = (ranks + 1).numpy()
    bins = np.unique(np.logspace(0, math.log10(len(images)), 40).astype(int))
    left.hist(positions, bins=bins, color=BLUE)
    left.set_xscale("log")
    for step, k in enumerate(RECALL_KS):
        left.axvline(k + 0.5, color=GREY, linewidth=1)
        left.annotate(f"R@{k} {100 * (positions <= k).mean():.1f}%", (k + 0.5, 0.95 - 0.09 * step),
                      xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=9, va="top")
    left.set(xlabel="Rank of the own image (log scale)", ylabel="Captions",
             title=f"Where each caption's own image lands (median rank {int(np.median(positions))})")

    scores = captions @ images.T
    own = scores.gather(1, caption_images[:, None].to(scores.device)).squeeze(1).cpu()
    others = scores.clone()
    others[torch.arange(len(scores)), caption_images] = torch.nan
    others = others[~others.isnan()]
    others = others[torch.randperm(len(others), device=others.device)[:200_000]].cpu()
    bins = np.linspace(min(others.min(), own.min()).item(), max(others.max(), own.max()).item(), 80)
    right.hist(others.numpy(), bins=bins, density=True, alpha=0.75, color=GREY, label="caption with another image")
    right.hist(own.numpy(), bins=bins, density=True, alpha=0.75, color=BLUE, label="caption with its own image")
    right.set(xlabel="Cosine similarity", ylabel="Density", title="Matching pairs score higher, but overlap")
    right.legend(frameon=False)
    figure.tight_layout()
    save(figure, path)


def plot_words(path, data, ranks):
    caption_words = [set(caption.upper().split()) for caption in data.captions]
    counts = Counter(word for words in caption_words for word in words if word not in STOPWORDS and word.isalpha())
    rows = []
    for word, count in counts.most_common(COMMON_WORDS):
        word_ranks = ranks[torch.tensor([word in words for words in caption_words])]
        rows.append((f"{word.lower()} ({count})", recall(word_ranks)))
    rows.sort(key=lambda row: row[1]["R@10"])
    figure, axis = plt.subplots(figsize=(10, 0.32 * len(rows) + 1.5))
    y = np.arange(len(rows))
    overall = recall(ranks)
    for k, offset, colour in ((10, 0.2, AQUA), (1, -0.2, BLUE)):
        axis.barh(y + offset, [row[1][f"R@{k}"] for row in rows], 0.38, color=colour, label=f"R@{k}")
        axis.axvline(overall[f"R@{k}"], color=colour, linewidth=1, linestyle=":")
    axis.set_yticks(y, [row[0] for row in rows])
    axis.set(xlabel="Caption → image recall (%) of the captions containing the word (dotted: all captions)",
             title=f"The {COMMON_WORDS} most common words in the val captions (number of captions)")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    save(figure, path)


def evaluate_model(image_model):
    folder = evaluation_dir(image_model)
    val = Split("val")
    train = Split("train").subset(len(val.image_ids), seed=SEED)
    report = {"model": str(image_model.dir), "epoch": image_model.epoch}
    for data in (val, train):
        images, captions = image_model.embed(data), caption_embeddings(data)
        ranks = caption_ranks(captions, images, data.caption_images)
        report[data.name] = {
            "caption_to_image": direction_metrics(ranks),
            "image_to_caption": direction_metrics(image_ranks(captions, images, data.caption_images)),
        }
        print(f"{data.name:9} caption->image R@1/5/10 {format_recall(report[data.name]['caption_to_image'])} | "
              f"image->caption {format_recall(report[data.name]['image_to_caption'])}")
        if data is val:
            plot_ranks(folder / "ranks.png", ranks, captions, images, data.caption_images)
            plot_words(folder / "words.png", data, ranks)

    crop_images = image_model.embed(val, crops=True)
    improved_ranks = caption_ranks(caption_embeddings(val), crop_images, val.caption_images, hubness(crop_images))
    report["val_improved_search"] = {"caption_to_image": direction_metrics(improved_ranks)}
    print(f"val with extra crops + hubness correction: caption->image R@1/5/10 "
          f"{format_recall(report['val_improved_search']['caption_to_image'])}")
    (folder / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def write_comparison(reports, path=COMPARISON_FILE):
    rows = [{"model": report["model"], "epoch": report["epoch"],
             **{f"{split} {direction} R@{k}": report[split][direction][f"R@{k}"]
                for split, direction in (("val", "caption_to_image"), ("val", "image_to_caption"),
                                         ("val_improved_search", "caption_to_image"))
                for k in RECALL_KS},
             **({"concept accuracy": round(report["concept_accuracy"], 1)} if "concept_accuracy" in report else {})}
            for report in reports]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nval caption -> image R@1/5/10 (also in {path}):")
    width = max(len(report["model"]) for report in reports)
    for report in sorted(reports, key=lambda report: -report["val"]["caption_to_image"]["R@1"]):
        train_split = next(key for key in report if key.startswith("train"))
        print(f"  {report['model']:{width}}  epoch {report['epoch']:>2}  {format_recall(report['val']['caption_to_image'])}"
              f"  improved search {format_recall(report['val_improved_search']['caption_to_image'])}"
              f"  train R@1 {report[train_split]['caption_to_image']['R@1']:.1f}"
              + (f"  concepts {report['concept_accuracy']:.1f}%" if "concept_accuracy" in report else ""))


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation of trained image CNNs on val.")
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_DIR], help=f"one or more models to compare: {MODEL_HELP}")
    reports = []
    for model_spec in parser.parse_args().model:
        image_model = load_model(model_spec)
        print(f"\n{image_model.label}", flush=True)
        reports.append(evaluate_model(image_model))
    if len(reports) > 1:
        write_comparison(reports)


if __name__ == "__main__":
    main()
