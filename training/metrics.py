"""Retrieval metrics from caption embeddings, image embeddings, and which image each caption belongs to."""

import torch

RECALL_KS = (1, 5, 10)
PRECISION_K = 10
# Left out of the class averages: people are in about 40% of the images and in every kind of scene, so the class says little about the model.
NOT_AVERAGED = {"PEOPLE"}


def recall(scores, is_match):
    # A hit if any match is among the query's top-k; is_match has the same shape as scores.
    hits = is_match.gather(1, scores.topk(max(RECALL_KS), dim=1).indices)
    return [100.0 * hits[:, :k].any(dim=1).float().mean().item() for k in RECALL_KS]


def effective_rank(embeddings):
    # exp(entropy) of the normalised singular values: how many dimensions the embeddings really use.
    # Near 1 means the embeddings have collapsed onto one direction; near the embedding size means they are spread out.
    singular_values = torch.linalg.svdvals((embeddings - embeddings.mean(dim=0)).cpu())
    shares = singular_values / singular_values.sum()
    return torch.exp(-(shares * shares.clamp_min(1e-12).log()).sum()).item()


def class_confusion(caption_embeddings, image_embeddings, caption_images, image_main_classes, class_count, k=PRECISION_K):
    # Row c: for the captions of the images of class c, the share (%) of their top-k images in each class.
    # image_main_classes holds one class index per image (class_count - 1 for OTHER).
    top_classes = image_main_classes[(caption_embeddings @ image_embeddings.T).topk(k, dim=1).indices]
    query_classes = image_main_classes[caption_images]
    matrix = torch.zeros(class_count, class_count)
    for index in range(class_count):
        retrieved = top_classes[query_classes == index].flatten()
        if len(retrieved):
            matrix[index] = 100.0 * torch.bincount(retrieved, minlength=class_count).float().cpu() / len(retrieved)
    return matrix


def retrieval_metrics(caption_embeddings, image_embeddings, caption_images, image_classes, class_names):
    scores = caption_embeddings @ image_embeddings.T
    is_match = caption_images[:, None] == torch.arange(len(image_embeddings), device=scores.device)
    metrics = {}

    # Caption -> image: the caption's own image must be in the top-k.
    for k, value in zip(RECALL_KS, recall(scores, is_match)):
        metrics[f"t2i R@{k}"] = value
    # Image -> caption: any of the image's captions in the top-k counts.
    for k, value in zip(RECALL_KS, recall(scores.T, is_match.T)):
        metrics[f"i2t R@{k}"] = value

    true_scores = scores.gather(1, caption_images[:, None])
    ranks = (scores > true_scores).sum(dim=1) + 1
    metrics["t2i MRR"] = (1.0 / ranks).mean().item()
    metrics["t2i median rank"] = ranks.float().median().item()

    # Mean cosine of matching pairs vs all other pairs; the gap between them is what training widens.
    metrics["t2i pos cos"] = true_scores.mean().item()
    metrics["t2i neg cos"] = ((scores.sum() - true_scores.sum()) / (scores.numel() - len(scores))).item()
    metrics["image effective rank"] = effective_rank(image_embeddings)

    # Class precision. image_classes is a boolean (images, classes) matrix, and an image can have several classes.
    # For the captions of the images of a class: the share of their top-k images that have that class too.
    # Lift divides it by the class's share of all images, which is what a random top-k would score: 1 is chance level.
    query_classes = image_classes[caption_images]
    top_share = image_classes[scores.topk(PRECISION_K, dim=1).indices].float().mean(dim=1)
    precisions, lifts = [], []
    for index, name in enumerate(class_names):
        queries = query_classes[:, index]
        if not queries.any():
            continue
        precision = top_share[queries, index].mean().item()
        lift = precision / image_classes[:, index].float().mean().item()
        metrics[f"class P@{PRECISION_K} {name}"] = 100.0 * precision
        metrics[f"class lift@{PRECISION_K} {name}"] = lift
        if name not in NOT_AVERAGED:
            precisions.append(100.0 * precision)
            lifts.append(lift)
    # Averaged over classes, so frequent classes do not hide rare ones.
    metrics[f"class P@{PRECISION_K}"] = sum(precisions) / len(precisions)
    metrics[f"class lift@{PRECISION_K}"] = sum(lifts) / len(lifts)
    return metrics
