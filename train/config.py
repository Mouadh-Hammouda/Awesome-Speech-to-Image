import argparse
import ast
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    seed: int = 53
    epochs: int = 80
    warmup_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1.0
    ema_decay: float = 0.99
    learn_temperature: bool = True
    initial_temperature: float = 0.07
    min_temperature: float = 0.01
    image_size: int = 224
    min_crop_area: float = 0.25
    grayscale_probability: float = 0.1
    erase_probability: float = 0.5
    erase_area: tuple = (0.02, 0.25)
    widths: tuple = (80, 160, 320, 640)
    depths: tuple = (2, 2, 2, 2)
    dropout: float = 0.5
    drop_path: float = 0.1
    checkpoint: bool = True
    results: Path = Path("results/image_cnn")

    def model_args(self):
        return {"widths": self.widths, "depths": self.depths, "dropout": self.dropout, "drop_path": self.drop_path}


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_config():
    settings = [field.name for field in fields(Config) if field.name != "results"]
    parser = argparse.ArgumentParser(description="Train the image CNN.")
    parser.add_argument("--results", type=Path, default=Config.results, help="output folder")
    parser.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                        help=f"override settings, e.g. --set epochs=40 seed=1 (any of: {', '.join(settings)})")
    arguments = parser.parse_args()

    overrides = {}
    for setting in arguments.set:
        name, _, text = setting.partition("=")
        if name.lower() not in settings:
            parser.error(f"unknown setting {name!r}")
        overrides[name.lower()] = parse_value(text)
    config = Config(**overrides, results=arguments.results)

    if config.warmup_epochs >= config.epochs:
        parser.error("warmup_epochs must be smaller than epochs")
    return config
