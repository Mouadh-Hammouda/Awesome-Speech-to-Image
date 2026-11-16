import argparse
import json
import matplotlib.pyplot as plt
import torch
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, Split, evaluation_dir, load_model
from evaluate.concepts import NAMES, NO_CONCEPT, concept_anchors, concept_labels
from evaluate.figures import save

TOP_CONFUSIONS = 5


def confusion(labels, predictions):
    matrix = torch.zeros(len(NAMES), len(NAMES), dtype=torch.long)
    known = labels != NO_CONCEPT
    for true, predicted in zip(labels[known].tolist(), predictions[known].tolist()):
        matrix[true, predicted] += 1
    return matrix


def most_frequent_confusions(matrix):
    mistakes = matrix.clone().fill_diagonal_(0).flatten()
    confusions = []
    for index in mistakes.argsort(descending=True)[:TOP_CONFUSIONS].tolist():
        if mistakes[index]:
            confusions.append({"captions_say": NAMES[index // len(NAMES)], "cnn_says": NAMES[index % len(NAMES)],
                               "images": mistakes[index].item()})
    return confusions


def plot(path, matrix, accuracy):
    shares = matrix.float() / matrix.sum(1, keepdim=True).clamp(min=1)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    axis.grid(False)
    for true, predicted in torch.nonzero(matrix).tolist():
        axis.text(predicted, true, matrix[true, predicted].item(), ha="center", va="center", fontsize=9,
                  color="white" if shares[true, predicted] > 0.55 else "black")
    axis.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(NAMES)), [f"{name} ({count})" for name, count in zip(NAMES, matrix.sum(1).tolist())])
    axis.set(xlabel="Concept the CNN's embedding is closest to", ylabel="Concept in the image's captions (images)",
             title=f"Zero-shot concepts on val: {accuracy:.1f}% correct over {matrix.sum().item()} images")
    figure.tight_layout()
    save(figure, path)


def evaluate_concepts(image_model, anchors):
    folder = evaluation_dir(image_model)
    val = Split("val")
    predictions = (image_model.embed(val) @ anchors.T).argmax(1).cpu()
    matrix = confusion(concept_labels(val), predictions)
    accuracy = 100 * matrix.trace().item() / max(matrix.sum().item(), 1)
    confusions = most_frequent_confusions(matrix)

    plot(folder / "confusion_matrix.png", matrix, accuracy)
    per_concept = {name: round(100 * matrix[index, index].item() / max(matrix[index].sum().item(), 1), 1)
                   for index, name in enumerate(NAMES)}
    (folder / "confusion_matrix.json").write_text(json.dumps({
        "accuracy": round(accuracy, 2), "images": matrix.sum().item(), "accuracy_per_concept": per_concept,
        "most_frequent_confusions": confusions, "concepts": NAMES, "counts": matrix.tolist(),
    }, indent=2))
    print(f"concept accuracy {accuracy:.1f}% over {matrix.sum().item()} val images")
    for item in confusions:
        print(f"  {item['images']:3} × captions say {item['captions_say']}, CNN says {item['cnn_says']}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Zero-shot confusion matrix of a trained image CNN on val.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help=MODEL_HELP)
    image_model = load_model(parser.parse_args().model)
    print(image_model.label, flush=True)
    evaluate_concepts(image_model, concept_anchors())


if __name__ == "__main__":
    main()
