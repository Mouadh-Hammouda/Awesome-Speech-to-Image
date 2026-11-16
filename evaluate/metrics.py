import torch

RECALL_KS = (1, 5, 10)


def search_scores(captions, images, image_hubness=None):
    scores = captions @ images.T
    return scores if image_hubness is None else 2 * scores - image_hubness[None]


def caption_ranks(captions, images, caption_images, image_hubness=None, chunk_size=4096):
    ranks = []
    for start in range(0, len(captions), chunk_size):
        scores = search_scores(captions[start:start + chunk_size], images, image_hubness)
        # The own score must come from the same matrix: computed another way, rounding can rank an image below itself.
        own = scores.gather(1, caption_images[start:start + chunk_size, None].to(scores.device))
        ranks.append((scores > own).sum(1))
    return torch.cat(ranks).cpu()


def image_ranks(captions, images, caption_images, chunk_size=4096):
    caption_images = caption_images.to(images.device)
    chunks = [captions[start:start + chunk_size] @ images.T for start in range(0, len(captions), chunk_size)]
    own = torch.cat([scores.gather(1, caption_images[start:start + chunk_size, None]).squeeze(1)
                     for start, scores in zip(range(0, len(captions), chunk_size), chunks)])
    best_own = torch.full((len(images),), -torch.inf, device=images.device)
    best_own = best_own.scatter_reduce(0, caption_images, own, reduce="amax")
    return sum((scores > best_own[None]).sum(0) for scores in chunks).cpu()


def recall(ranks):
    return {f"R@{k}": 100 * (ranks < k).float().mean().item() for k in RECALL_KS}
