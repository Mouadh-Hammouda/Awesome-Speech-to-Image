import argparse
from pathlib import Path
from evaluate.common import DEFAULT_MODEL_DIR, MODEL_HELP, load_model
from evaluate.concepts import concept_anchors
from evaluate.confusion_matrix import evaluate_concepts
from evaluate.examples import plot_all_examples
from evaluate.recall import evaluate_model, write_comparison
from evaluate.semantic_map import draw_map

REPORT_FILE = Path("results/evaluation_report.csv")


def main():
    parser = argparse.ArgumentParser(description="Recall, confusion matrix, semantic map and examples of each model.")
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_DIR], help=MODEL_HELP)
    anchors = concept_anchors()
    reports = []
    for model_spec in parser.parse_args().model:
        image_model = load_model(model_spec)
        print(f"\n{image_model.label}", flush=True)
        report = evaluate_model(image_model)
        report["concept_accuracy"] = evaluate_concepts(image_model, anchors)
        draw_map(image_model, show=False)
        plot_all_examples(image_model)
        reports.append(report)
    write_comparison(reports, REPORT_FILE)


if __name__ == "__main__":
    main()
