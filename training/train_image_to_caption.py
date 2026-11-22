import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # must be set before torch is imported

import csv
import math
import sys
import time
from pathlib import Path

# Allow direct execution from the repository root: python3 training/train_image_to_caption.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from data_processing.augment import augment, resize
from data_processing.caption_data import CaptionSplit
from common import CLASSES, EMBEDDING_SIZE, IMAGE_SIZE, get_device, model_file
from retrieval.improved_search import embed_for_search, hubness, search_scores
from training.console import TrainingDisplay, print_block
from training.metrics import RECALL_KS, class_confusion, recall, retrieval_metrics
from tools.plot_training import plot_confusion, plot_runs
from models.image_cnn import DILATIONS, ImageEncoderCNN, saved_dilations

EPOCHS = 80
# The learning rate rises linearly over these epochs, then follows a cosine curve down to zero.
WARMUP_EPOCHS = 2
# Images per batch; each comes with all of its captions (about 5), so a batch has about 5 times as many captions.
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
# AdamW's weight decay is applied separately from the gradient step; it is not applied to biases and normalisation weights.
WEIGHT_DECAY = 1.0
# The saved and evaluated model is a moving average of the weights over recent steps, which is steadier than the latest weights.
EMA_DECAY = 0.99
# Scores are cosine similarities divided by a temperature, learned during training and kept at or above the minimum.
INITIAL_TEMPERATURE = 0.07
MIN_TEMPERATURE = 0.01
# Soft targets: other images in the batch whose captions mean nearly the same as a caption are not counted as fully wrong.
# The target is this share of a "teacher" distribution, from the text model's caption similarities, and the rest on the caption's own image.
# 0 gives plain one-hot targets. With 0.5 about 16% of the target goes to other images on average, more only where they fit as well.
# With an earlier ResNet-style CNN, 0.5 narrowed the train/val gap but lowered val R@1 (17.9 vs 18.2), so it is off.
SOFT_TARGET_WEIGHT = 0.0
# How sharply the teacher prefers the most similar images; in the full 1024-d space caption cosines are about 0.66 to the same image's other captions and 0 to unrelated ones.
SOFT_TARGET_TEMPERATURE = 0.05
# Train images (and their captions) scored each epoch, to compare train and val recall; about the size of val.
TRAIN_EVAL_IMAGES = 2000
CLASS_NAMES = list(CLASSES)
# The val metric that decides which epoch is saved as the best model.
BEST_METRIC = "t2i R@1"

torch.manual_seed(53)
device = get_device()


def weight_shapes(state):
    return {name: value.shape for name, value in state.items()}


def make_optimizer(model, logit_scale):
    no_decay = {name for name, parameter in model.named_parameters() if parameter.ndim < 2}
    return torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters() if name not in no_decay], "weight_decay": WEIGHT_DECAY},
        {"params": [p for name, p in model.named_parameters() if name in no_decay], "weight_decay": 0.0},
        {"params": [logit_scale], "weight_decay": 0.0},
    ], lr=LEARNING_RATE)


def make_scheduler(optimizer, steps_per_epoch):
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = EPOCHS * steps_per_epoch

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    # Stepped once per batch.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def soft_targets(captions, caption_images, image_count):
    # How well each batch image's captions match each caption, in the frozen text model's space.
    caption_sums = torch.zeros(image_count, captions.shape[1], device=captions.device).index_add_(0, caption_images, captions)
    similarity = captions @ F.normalize(caption_sums, dim=-1).T
    # For its own image, a caption is compared with that image's other captions only, not with itself.
    rows = torch.arange(len(captions), device=captions.device)
    similarity[rows, caption_images] = (captions * F.normalize(caption_sums[caption_images] - captions, dim=-1)).sum(dim=-1)
    teacher = (similarity / SOFT_TARGET_TEMPERATURE).softmax(dim=1)
    return SOFT_TARGET_WEIGHT * teacher + (1 - SOFT_TARGET_WEIGHT) * F.one_hot(caption_images, image_count).float()


def caption_to_image_loss(image_embeddings, captions, caption_images, logit_scale):
    # Each caption (a row of scores) must pick its own image out of the batch's images; the other direction is not trained.
    # Returns the training loss (cross-entropy with soft targets) and the plain one-hot loss, which is logged for comparison.
    scores = logit_scale.exp() * captions @ image_embeddings.T
    log_probabilities = scores.log_softmax(dim=1)
    loss = -(soft_targets(captions, caption_images, len(image_embeddings)) * log_probabilities).sum(dim=1).mean()
    return loss, F.nll_loss(log_probabilities, caption_images)


def train_epoch(model, ema_model, logit_scale, optimizer, scheduler, scaler, data, display):
    model.train()
    total_loss, total_hard_loss, steps = 0.0, 0.0, 0
    for images in torch.randperm(len(data)).split(BATCH_SIZE):
        # A smaller last batch would have fewer negatives than the others.
        if len(images) < BATCH_SIZE:
            continue
        batch_images = augment(data.load_images(images).to(device, non_blocking=True))
        captions, caption_images = data.batch_captions(images)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            image_embeddings = model(batch_images)
        loss, hard_loss = caption_to_image_loss(image_embeddings.float(), captions.to(device), caption_images.to(device), logit_scale)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema_model.update_parameters(model)
        with torch.no_grad():
            logit_scale.clamp_(max=math.log(1 / MIN_TEMPERATURE))
        total_loss += loss.item()
        total_hard_loss += hard_loss.item()
        steps += 1
        display.batch_done(total_loss / steps)
    return total_loss / steps, total_hard_loss / steps


@torch.no_grad()
def embed_images(model, data, image_count):
    model.eval()
    embeddings = []
    for indices in torch.arange(image_count).split(BATCH_SIZE):
        batch = resize(data.load_images(indices).to(device))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            # An image and its mirror image give one embedding together.
            embedding = model(batch).float() + model(batch.flip(-1)).float()
        embeddings.append(F.normalize(embedding, dim=-1))
    return torch.cat(embeddings)


@torch.no_grad()
def batched_loss(image_embeddings, captions, caption_images, logit_scale):
    # The training loss without augmentation: batches of BATCH_SIZE images in order, each with all of its captions, as in training.
    # The last, smaller batch is left out, since fewer images make the task easier.
    losses = []
    for start in range(0, len(image_embeddings) - BATCH_SIZE + 1, BATCH_SIZE):
        rows = ((caption_images >= start) & (caption_images < start + BATCH_SIZE)).nonzero().squeeze(1)
        loss, _ = caption_to_image_loss(image_embeddings[start:start + BATCH_SIZE], captions[rows], caption_images[rows] - start, logit_scale)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, data, logit_scale, max_images=None):
    # max_images scores only the first images and their captions, which keeps the train check quick.
    image_count = max_images or len(data)
    caption_rows = (data.caption_images < image_count).nonzero().squeeze(1)
    captions, caption_images = data.captions[caption_rows].to(device), data.caption_images[caption_rows].to(device)
    image_embeddings = embed_images(model, data, image_count)
    metrics = retrieval_metrics(captions, image_embeddings, caption_images, data.image_classes[:image_count].to(device), CLASS_NAMES)
    metrics["loss"] = batched_loss(image_embeddings, captions, caption_images, logit_scale)
    return metrics


def load_best_model(model_path):
    saved = torch.load(model_path, map_location=device)
    model = ImageEncoderCNN(saved_dilations(saved)).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(saved["model"])
    return model, saved["metrics"]


@torch.no_grad()
def improved_search_recall(model, data, train_captions):
    # Recall as the search window ranks images. It is only reported: the best epoch is still chosen by the plain val recall.
    images = embed_for_search(model, data, device)
    scores = search_scores(data.captions.to(device), images, hubness(images, train_captions))
    is_match = data.caption_images.to(device)[:, None] == torch.arange(len(images), device=device)
    return recall(scores, is_match)


@torch.no_grad()
def save_confusion_matrix(model, metrics, data, output_file):
    # Which classes the val captions of each class retrieve with the saved best model; OTHER is an image without one clear class.
    class_names = [*CLASSES, "OTHER"]
    matrix = class_confusion(data.captions.to(device), embed_images(model, data, len(data)), data.caption_images.to(device), data.image_main_classes.to(device), len(class_names))
    plot_confusion(matrix, class_names, output_file, f"val: classes of each caption's top-10 images, best model (epoch {metrics['epoch']}, t2i R@1 {metrics['t2i R@1']:.1f})")


def log_metrics(log_file, row):
    is_new = not log_file.exists()
    with open(log_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        # Significant digits rather than decimal places, so small values such as the learning rate keep their precision.
        writer.writerow({key: value if isinstance(value, int) else float(f"{value:.6g}") for key, value in row.items()})


def main():
    output_file = model_file()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print("Loading data...", end="\r", flush=True)
    train_data, val_data = CaptionSplit("train"), CaptionSplit("val")

    model = ImageEncoderCNN().to(device).to(memory_format=torch.channels_last)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
    # Learned as a log, so the temperature stays positive.
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / INITIAL_TEMPERATURE), device=device))
    optimizer = make_optimizer(model, logit_scale)
    steps_per_epoch = len(train_data) // BATCH_SIZE
    scheduler = make_scheduler(optimizer, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print_block("Training the image CNN", {
        "images": f"{len(train_data):,} train, {len(val_data):,} val",
        "model": f"{sum(p.numel() for p in model.parameters()):,} parameters, {EMBEDDING_SIZE}-d output, on {device}",
        "dilations": DILATIONS,
        "folder": f"{output_file.parent}/",
    })

    # The latest training state, saved every epoch so an interrupted run picks up where it left off.
    checkpoint_file = output_file.with_name("checkpoint.pt")
    best_recall = -1.0
    start_epoch = 1
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "ema_model" not in checkpoint or weight_shapes(checkpoint["model"]) != weight_shapes(model.state_dict()):
            raise SystemExit(f"{checkpoint_file} is from another model or an older version of this script; move it (and the model next to it) away to start a new run.")
        if saved_dilations(checkpoint) != DILATIONS:
            raise SystemExit(f"{checkpoint_file} was trained with dilations {saved_dilations(checkpoint)}, but models/image_cnn.py now has {DILATIONS}; set them back to resume, or move the folder away to start a new run.")
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        logit_scale.data.fill_(checkpoint["logit_scale"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_recall = checkpoint["best_recall"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch <= EPOCHS:
            print(f"Resuming at epoch {start_epoch}, best val R@1 so far: {best_recall:.1f}")

    run_start = time.monotonic()
    if start_epoch <= EPOCHS:
        print("recall = % of captions that find their image in the top k")
        print("median rank = position of the right image, lower is better")
        print("* = new best val recall@1, model saved")
    with TrainingDisplay(EPOCHS, start_epoch, steps_per_epoch) as display:
        for epoch in range(start_epoch, EPOCHS + 1):
            display.start_epoch(epoch)
            epoch_start = time.monotonic()
            train_loss, hard_loss = train_epoch(model, ema_model, logit_scale, optimizer, scheduler, scaler, train_data, display)
            display.set_stage("eval")
            # The averaged weights are what is evaluated and saved.
            val = evaluate(ema_model.module, val_data, logit_scale)
            train = evaluate(ema_model.module, train_data, logit_scale, TRAIN_EVAL_IMAGES)
            epoch_seconds = time.monotonic() - epoch_start
            learning_rate = scheduler.get_last_lr()[0]
            temperature = 1 / logit_scale.exp().item()
            log_metrics(output_file.with_name("training_metrics.csv"), {"epoch": epoch, "seconds": epoch_seconds, "lr": learning_rate, "loss": train_loss, "one-hot loss": hard_loss, "temperature": temperature, "train eval loss": train["loss"], "train t2i R@1": train["t2i R@1"], "train t2i R@10": train["t2i R@10"], **{f"val {name}": value for name, value in val.items()}})
            # Redrawn every epoch, so training_curves.png can be opened while the run is going.
            plot_runs([output_file.parent], output_file.with_name("training_curves.png"))
            is_best = val[BEST_METRIC] > best_recall
            if is_best:
                best_recall = val[BEST_METRIC]
                torch.save({"model": ema_model.module.state_dict(), "dilations": DILATIONS, "image_size": IMAGE_SIZE, "metrics": {"epoch": epoch, **val}}, output_file)
            torch.save({"model": model.state_dict(), "dilations": DILATIONS, "ema_model": ema_model.state_dict(), "logit_scale": logit_scale.item(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_recall": best_recall}, checkpoint_file)
            display.epoch_done(epoch_seconds, train_loss, train, val, is_best)

    # Also made when the script is run again on a finished run, which only loads the checkpoint and skips training.
    confusion_file = output_file.with_name("val_confusion_matrix.png")
    best_model, best = load_best_model(output_file)
    save_confusion_matrix(best_model, best, val_data, confusion_file)
    improved = improved_search_recall(best_model, val_data, train_data.captions)
    print()
    title = f"Finished in {(time.monotonic() - run_start) / 60:.0f} min" if start_epoch <= EPOCHS else "Already finished, nothing to train"
    print_block(title, {
        "best epoch": best["epoch"],
        "val recall": "   ".join(f"@{k} {best[f't2i R@{k}']:.1f}" for k in RECALL_KS),
        "with search": "   ".join(f"@{k} {value:.1f}" for k, value in zip(RECALL_KS, improved)),
        "model": output_file,
        "curves": output_file.with_name("training_curves.png"),
        "confusion": confusion_file,
    })


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue from the last saved epoch.")
        raise SystemExit(130)
