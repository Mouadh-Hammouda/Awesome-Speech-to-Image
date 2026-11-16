import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # must be set before importing torch

import csv
import json
import math
import time
from dataclasses import asdict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from evaluate.metrics import RECALL_KS, caption_ranks, image_ranks, recall
from models.image_cnn import ImageCNN
from train.augment import augment
from train.config import parse_config
from train.data import CachedSplit
from train.loss import info_nce
from utils import get_device

device = get_device()


def make_optimizer(model, logit_scale, config):
    no_decay = {name for name, parameter in model.named_parameters() if parameter.ndim < 2 or "position" in name}
    groups = [
        {"params": [p for name, p in model.named_parameters() if name not in no_decay], "weight_decay": config.weight_decay},
        {"params": [p for name, p in model.named_parameters() if name in no_decay], "weight_decay": 0.0},
    ]
    if logit_scale.requires_grad:
        groups.append({"params": [logit_scale], "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def make_scheduler(optimizer, steps_per_epoch, config):
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, ema_model, logit_scale, optimizer, scheduler, scaler, data, config):
    model.train()
    total_loss, steps = 0.0, 0
    for images in torch.randperm(len(data)).split(config.batch_size):
        if len(images) < config.batch_size:
            continue
        batch_images = augment(data.load_images(images).to(device, non_blocking=True), config)
        captions, caption_images = data.batch_captions(images)
        caption_images = caption_images.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            image_embeddings = model(batch_images)
        loss = info_nce(image_embeddings.float(), captions.to(device), caption_images, logit_scale)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema_model.update_parameters(model)
        with torch.no_grad():
            logit_scale.clamp_(max=math.log(1 / config.min_temperature))
        total_loss += loss.item()
        steps += 1
    return total_loss / steps


@torch.no_grad()
def embed_images(model, data, batch_size):
    model.eval()
    embeddings = []
    for indices in torch.arange(len(data)).split(batch_size):
        batch = data.load_images(indices).to(device).float().contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embedding = model(batch).float() + model(batch.flip(-1)).float()
        embeddings.append(F.normalize(embedding, dim=-1))
    return torch.cat(embeddings)


@torch.no_grad()
def evaluate(model, data, batch_size):
    images = embed_images(model, data, batch_size)
    captions = data.captions.to(device)
    metrics = {}
    for direction, ranks in (("c2i", caption_ranks(captions, images, data.caption_images)),
                             ("i2c", image_ranks(captions, images, data.caption_images))):
        metrics |= {f"{direction}_{name}": value for name, value in recall(ranks).items()}
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
    return metrics


def format_metrics(metrics):
    return " | ".join(f"{label} R@1/5/10 " + "/".join(f"{metrics[f'{direction}_R@{k}']:.1f}" for k in RECALL_KS)
                      for direction, label in (("c2i", "caption->image"), ("i2c", "image->caption")))


def save_history(history, folder):
    with open(folder / "training_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, recall_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis.plot(epochs[1:], [row["loss"] for row in history[1:]])
    loss_axis.set(title="Training loss", xlabel="Epoch", ylabel="InfoNCE")
    for k in RECALL_KS:
        recall_axis.plot(epochs, [row[f"c2i_R@{k}"] for row in history], label=f"caption → image R@{k}")
    recall_axis.set(title="Validation recall (EMA model)", xlabel="Epoch", ylabel="%")
    recall_axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "training_curves.png", dpi=120)
    plt.close(figure)


def main():
    config = parse_config()
    torch.manual_seed(config.seed)
    config.results.mkdir(parents=True, exist_ok=True)
    (config.results / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))
    print("Loading data...", flush=True)
    train_data, val_data = CachedSplit("train", config.image_size), CachedSplit("val", config.image_size)

    model = ImageCNN(output_size=train_data.captions.shape[1], **config.model_args())
    model.checkpoint_blocks = config.checkpoint
    model = model.to(device).to(memory_format=torch.channels_last)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / config.initial_temperature), device=device),
                                     requires_grad=config.learn_temperature)
    optimizer = make_optimizer(model, logit_scale, config)
    scheduler = make_scheduler(optimizer, len(train_data) // config.batch_size, config)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}", flush=True)

    history, best = [], None
    for epoch in range(config.epochs + 1):
        start_time = time.monotonic()
        loss = math.nan
        if epoch:
            loss = train_epoch(model, ema_model, logit_scale, optimizer, scheduler, scaler, train_data, config)
        metrics = evaluate(ema_model.module, val_data, config.batch_size)
        temperature = 1 / logit_scale.exp().item()
        history.append({"epoch": epoch, "loss": loss, "temperature": temperature, **metrics})
        save_history(history, config.results)
        print(f"epoch {epoch} | loss {loss:.3f} | temperature {temperature:.3f} | {format_metrics(metrics)} | "
              f"{time.monotonic() - start_time:.0f}s", flush=True)
        if best is None or metrics["mean_recall"] > best["mean_recall"]:
            best = history[-1]
            torch.save({"model": ema_model.module.state_dict(), "model_args": config.model_args(),
                        "image_size": config.image_size, "metrics": best}, config.results / "model.pt")
            print(f"saved best model (mean recall {best['mean_recall']:.1f})", flush=True)
    print(f"best epoch {best['epoch']} | {format_metrics(best)} | {config.results / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
