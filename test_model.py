import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import (
    CAPTION_DEVICE, CAPTION_MODEL_NAME, CHECKPOINT_PATH, GUI_PORT,
    RETRIEVAL_SPLIT, RETRIEVAL_TOP_K,
)
from utils import get_image_path, load_image_gallery


def retrieve_images(caption, image_ids, image_features, encoder):
    query = encoder.encode(caption, normalize_embeddings=True, convert_to_tensor=True).cpu()
    scores = image_features @ query
    values, indices = scores.topk(min(RETRIEVAL_TOP_K, len(image_ids)))
    return [(image_ids[index], score) for index, score in zip(indices.tolist(), values.tolist())]


class CaptionRetriever:
    def __init__(self):
        self.dataset = None
        self.image_features = None
        self.encoder = None
        self.lock = threading.Lock()

    def search(self, caption):
        with self.lock:
            if self.dataset is None:
                if not CHECKPOINT_PATH.exists():
                    raise FileNotFoundError(f"Train the model first. Missing checkpoint: {CHECKPOINT_PATH}")
                self.dataset, self.image_features = load_image_gallery(RETRIEVAL_SPLIT)
            if self.encoder is None:
                self.encoder = SentenceTransformer(CAPTION_MODEL_NAME, device=CAPTION_DEVICE)
            return retrieve_images(caption, self.dataset.image_ids, self.image_features, self.encoder)


class GUIHandler(BaseHTTPRequestHandler):
    def send_content(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data, status=200):
        self.send_content(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        if self.path == "/":
            page = Path(__file__).with_name("model_gui.html").read_bytes()
            self.send_content(page, "text/html; charset=utf-8")
        elif self.path.startswith("/image/"):
            image_id = self.path.removeprefix("/image/")
            dataset = self.server.retriever.dataset
            if dataset is None or image_id not in dataset.image_ids:
                self.send_error(404)
                return
            path = get_image_path(RETRIEVAL_SPLIT, image_id)
            self.send_content(path.read_bytes(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/search":
            self.send_error(404)
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            caption = request["caption"].strip()
            if not caption:
                self.send_json({"error": "Write a caption first."}, 400)
                return
            results = self.server.retriever.search(caption)
            self.send_json({"results": [{"image_id": image_id, "score": score} for image_id, score in results]})
        except Exception as error:
            self.send_json({"error": str(error)}, 500)


def main():
    address = ("127.0.0.1", GUI_PORT)
    with ThreadingHTTPServer(address, GUIHandler) as server:
        server.retriever = CaptionRetriever()
        url = f"http://127.0.0.1:{GUI_PORT}"
        print(f"Caption search: {url}\nPress Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
