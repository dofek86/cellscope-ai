"""Read-only public demonstration of the frozen CellScope DINOv2 candidate.

This server deliberately omits the local review queue and never retains uploads.
It must listen on loopback behind a TLS reverse proxy with request-size limits.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .dino_predict import DinoCandidate
from .relative_texture_review import CELL_TYPES

Image.MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGES = 5
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 55 * 1024 * 1024
RATE_WINDOW_SECONDS = 10 * 60
RATE_MAX_REQUESTS = 8
INFERENCE = threading.BoundedSemaphore(1)
RATE_LOCK = threading.Lock()
RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
MODEL: DinoCandidate | None = None

PAGE = Path(__file__).with_name("public_demo_page.html").read_text(encoding="utf-8")


def _allow_request(ip: str) -> bool:
    now = time.monotonic()
    with RATE_LOCK:
        bucket = RATE_BUCKETS[ip]
        while bucket and now - bucket[0] > RATE_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_MAX_REQUESTS:
            return False
        bucket.append(now)
        return True


def _decode_image(entry: dict, destination: Path) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("data"), str):
        raise ValueError("Invalid image data.")
    encoded = entry["data"].split(",", 1)
    if len(encoded) != 2 or not encoded[0].startswith("data:image/"):
        raise ValueError("Only image uploads are accepted.")
    try:
        raw = base64.b64decode(encoded[1], validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid image encoding.") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Each image must be 8 MB or smaller.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "TIFF", "BMP"}:
                raise ValueError("Unsupported image format.")
            if image.width > 4096 or image.height > 4096 or image.width * image.height > 20_000_000:
                raise ValueError("Image dimensions are too large.")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This is not a supported image.") from exc
    destination.write_bytes(raw)
    name = entry.get("name", "image")
    return Path(str(name)[:160]).name


class Handler(BaseHTTPRequestHandler):
    server_version = "CellScope"

    def log_message(self, *_args):
        # The reverse proxy keeps basic request logs; do not log image names or data.
        pass

    def _reply(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' blob: data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, value: dict):
        self._reply(status, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/":
            self._reply(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200 if MODEL is not None else 503, {"status": "ready" if MODEL else "starting"})
        else:
            self._reply(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/predict":
            self._reply(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("Request is too large.")
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                raise ValueError("Expected JSON input.")
            ip = self.headers.get("X-Real-IP", self.client_address[0])
            if not _allow_request(ip):
                self._json(429, {"error": "Too many requests; try again in ten minutes."})
                return
            if not INFERENCE.acquire(blocking=False):
                self._json(503, {"error": "The model is busy. Please retry shortly."})
                return
            try:
                request = json.loads(self.rfile.read(size))
                images = request.get("images")
                if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
                    raise ValueError("Select one to five images from one acquisition session.")
                if "save_for_review" in request:
                    raise ValueError("The public demo does not store images for review.")
                session = request.get("session_id", "")
                cell_type = request.get("cell_type", "")
                if not isinstance(session, str) or len(session) > 100 or not isinstance(cell_type, str):
                    raise ValueError("Invalid session or cell type.")
                session, cell_type = session.strip(), cell_type.strip()
                if cell_type and cell_type not in CELL_TYPES:
                    raise ValueError("Unknown cell type.")
                if cell_type and len(images) >= 5 and not session:
                    raise ValueError("Enter a session ID for the optional batch signal.")
                with tempfile.TemporaryDirectory(prefix="cellscope-public-") as directory:
                    paths = []
                    names = []
                    for index, entry in enumerate(images):
                        path = Path(directory) / f"{index:02d}.image"
                        names.append(_decode_image(entry, path))
                        paths.append(path)
                    groups = [session or "uploaded_session"] * len(paths)
                    results = MODEL.predict_paths(paths, groups, cell_type=cell_type) if cell_type else MODEL.predict_paths(paths, groups)
                    for item, name in zip(results, names, strict=True):
                        item["image"] = name
                self._json(200, {"results": results})
            finally:
                INFERENCE.release()
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})


def main():
    global MODEL
    MODEL = DinoCandidate(device="cpu")
    port = int(os.environ.get("CELLSCOPE_PUBLIC_PORT", "7863"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CellScope public demo ready on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
