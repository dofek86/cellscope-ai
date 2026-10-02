"""Public demo boundaries: live inference without exposing the review queue."""

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image

from cellscope_ai import public_demo


class FakeModel:
    def __init__(self):
        self.upload_path = None

    def predict_paths(self, paths, _groups, cell_type=None):
        self.upload_path = paths[0]
        assert self.upload_path.exists()
        return [{"image": str(path), "probability_good": .42, "decision": "bad",
                 "texture_review_flag": None, "texture_review_stability": None}
                for path in paths]


def test_public_demo_does_not_expose_review_or_retain_uploads():
    model = FakeModel()
    public_demo.MODEL = model
    public_demo.RATE_BUCKETS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), public_demo.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for route in ("/review", "/api/review", "/api/review/image/test"):
            try:
                urllib.request.urlopen(root + route)
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError(f"Unexpected public review route: {route}")

        image = Image.new("RGB", (64, 64), "gray")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        payload = {"images": [{"name": "sample.png", "data": data_uri}], "session_id": "", "cell_type": ""}

        denied = urllib.request.Request(root + "/api/predict",
                                        data=json.dumps({**payload, "save_for_review": True}).encode(),
                                        headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(denied)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("The public demo accepted a storage request")

        request = urllib.request.Request(root + "/api/predict", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        assert result["results"][0]["image"] == "sample.png"
        assert result["results"][0]["probability_good"] == .42
        assert not model.upload_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
