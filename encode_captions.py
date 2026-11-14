import numpy as np
from config import (
    CAPTION_BATCH_SIZE, CAPTION_DEVICE, CAPTION_MODEL_NAME,
    DATA_DIR, SPLITS, TEXT_EMBEDDING_SIZE,
)
from utils import load_metadata


def encode_captions(encoder, split):
    rows = load_metadata(split)
    captions = [row["caption"] for row in rows]

    embeddings = encoder.encode(
        captions,
        batch_size=CAPTION_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(DATA_DIR / f"{split}_bge_m3.npy", embeddings)
    print(f"{split}: saved {embeddings.shape}")


def ensure_caption_embeddings():
    splits_to_encode = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}_bge_m3.npy"
        expected_shape = (len(load_metadata(split)), TEXT_EMBEDDING_SIZE)
        if not path.exists() or np.load(path, mmap_mode="r").shape != expected_shape:
            splits_to_encode.append(split)

    if splits_to_encode:
        print(f"Preparing caption embeddings: {', '.join(splits_to_encode)}")
        main(splits_to_encode)


def main(splits=SPLITS):
    from sentence_transformers import SentenceTransformer

    text_encoder = SentenceTransformer(
        CAPTION_MODEL_NAME,
        device=CAPTION_DEVICE,
    )
    for split in splits:
        encode_captions(text_encoder, split)


if __name__ == "__main__":
    main()
