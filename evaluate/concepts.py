from collections import Counter
import torch
import torch.nn.functional as F
from evaluate.common import Split, caption_embeddings

CONCEPTS = {
    "dog": {"DOG", "DOGS", "PUPPY", "PUPPIES"},
    "cat": {"CAT", "CATS", "KITTEN", "KITTENS", "KITTY"},
    "horse": {"HORSE", "HORSES"},
    "elephant": {"ELEPHANT", "ELEPHANTS"},
    "giraffe": {"GIRAFFE", "GIRAFFES"},
    "zebra": {"ZEBRA", "ZEBRAS"},
    "pizza": {"PIZZA", "PIZZAS"},
    "train": {"TRAIN", "TRAINS"},
    "bus": {"BUS", "BUSES"},
    "airplane": {"AIRPLANE", "AIRPLANES", "PLANE", "PLANES", "JET", "JETS"},
    "surfing": {"SURFBOARD", "SURFBOARDS", "SURFER", "SURFERS", "SURFING"},
    "skiing": {"SKI", "SKIS", "SKIER", "SKIERS", "SKIING"},
    "tennis": {"TENNIS"},
    "toilet": {"TOILET", "TOILETS"},
}
NAMES = list(CONCEPTS)
NO_CONCEPT = -1


def words(caption):
    return set(caption.upper().split())


def concept_labels(data):
    # An image's concept is the one most of its captions name, if at least two do and no other concept ties with it.
    labels = []
    for captions in data.image_captions:
        votes = Counter(index for caption in captions for index, concept_words in enumerate(CONCEPTS.values())
                        if words(caption) & concept_words)
        best = votes.most_common(2)
        clear = best and best[0][1] >= 2 and (len(best) == 1 or best[1][1] < best[0][1])
        labels.append(best[0][0] if clear else NO_CONCEPT)
    return torch.tensor(labels)


def concept_anchors(names=NAMES):
    train = Split("train")
    embeddings = caption_embeddings(train)
    caption_words = [words(caption) for caption in train.captions]
    all_concept_words = set().union(*CONCEPTS.values())
    masks = [[bool(caption & CONCEPTS[name]) and not caption & (all_concept_words - CONCEPTS[name]) for caption in caption_words]
             for name in names]
    return torch.stack([F.normalize(embeddings[torch.tensor(mask, device=embeddings.device)].mean(0), dim=0)
                        for mask in masks])
