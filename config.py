import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = Path(os.getenv("RAW_DIR", str(DATA_DIR / "raw")))
IMAGE_CACHE_DIR = Path(os.getenv("IMAGE_CACHE_DIR", str(DATA_DIR / "image_cache")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
SPLITS = ("train", "val")

SPOKENCOCO_URL = os.getenv("SPOKENCOCO_URL", "https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz")
COCO_IMAGES_URL = os.getenv("COCO_IMAGES_URL", "http://images.cocodataset.org")
# Use "Max" for either split to download every available pair.
MAX_PAIRS = {
    "train": os.getenv("MAX_TRAIN_PAIRS", "110000"),
    "val": os.getenv("MAX_VAL_PAIRS", "10000"),
}
IMAGE_DOWNLOAD_THREADS = int(os.getenv("IMAGE_DOWNLOAD_THREADS", "16"))
IMAGE_DOWNLOAD_ATTEMPTS = int(os.getenv("IMAGE_DOWNLOAD_ATTEMPTS", "3"))

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "128"))
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
TEXT_EMBEDDING_SIZE = int(os.getenv("TEXT_EMBEDDING_SIZE", "1024"))
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "BAAI/bge-m3")
CAPTION_DEVICE = os.getenv("CAPTION_DEVICE", "cpu")
CAPTION_BATCH_SIZE = int(os.getenv("CAPTION_BATCH_SIZE", "16"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "60"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
SEED = int(os.getenv("SEED", "53"))

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", str(RESULTS_DIR / "custom_cnn_best.pt")))
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
DEBUG_DIR = RESULTS_DIR / "debug"
DEBUG_SPLIT = os.getenv("DEBUG_SPLIT", "val")
# Keyword groups for the plots; these are approximate labels from captions.
DEBUG_CLASSES = tuple(name.strip() for name in os.getenv("DEBUG_CLASSES", "dog,cat,car").split(","))
# Embedding dimensions for the heatmap. PCA and retrieval use all dimensions.
DEBUG_FEATURES = tuple(int(index) for index in os.getenv("DEBUG_FEATURES", "0,1,2,3,4,5,6,7").split(","))
DEBUG_SAMPLES_PER_CLASS = int(os.getenv("DEBUG_SAMPLES_PER_CLASS", "100"))
RETRIEVAL_SPLIT = os.getenv("RETRIEVAL_SPLIT", "val")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
GUI_PORT = int(os.getenv("GUI_PORT", "7860"))
