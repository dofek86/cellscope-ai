"""Exercise the deployed HTTPS-facing proxy and real model with synthetic pixels.

Synthetic input checks only the plumbing, not biological accuracy.
"""

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

root = sys.argv[1].rstrip("/")
host = sys.argv[2] if len(sys.argv) > 2 else None
headers = {"Host": host} if host else {}


def get(path):
    with urllib.request.urlopen(urllib.request.Request(root + path, headers=headers), timeout=20) as response:
        return response.status, response.read()


status, page = get("/")
assert status == 200 and b"CellScope AI" in page
status, health = get("/health")
assert status == 200 and json.loads(health)["status"] == "ready"
try:
    get("/review")
except urllib.error.HTTPError as exc:
    assert exc.code == 404
else:
    raise AssertionError("Private review route is exposed")

pixels = Image.new("RGB", (256, 256), "gray")
buffer = io.BytesIO()
pixels.save(buffer, format="PNG")
payload = {"images": [{"name": "synthetic-smoke.png",
                       "data": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}],
           "session_id": "smoke", "cell_type": ""}
request = urllib.request.Request(root + "/api/predict", data=json.dumps(payload).encode(),
                                 headers={**headers, "Content-Type": "application/json"})
start = time.monotonic()
with urllib.request.urlopen(request, timeout=180) as response:
    prediction = json.load(response)["results"][0]
assert 0 <= prediction["probability_good"] <= 1
assert prediction["image"] == "synthetic-smoke.png"

batch = {"images": [dict(payload["images"][0], name=f"synthetic-{index}.png")
                    for index in range(5)],
         "session_id": "synthetic-smoke-session", "cell_type": "A549"}
request = urllib.request.Request(root + "/api/predict", data=json.dumps(batch).encode(),
                                 headers={**headers, "Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    batch_results = json.load(response)["results"]
assert len(batch_results) == 5
assert all(0 <= item["probability_good"] <= 1 for item in batch_results)
print("smoke_ok", "latency_seconds", round(time.monotonic() - start, 2),
      "page_bytes", len(page), "review_status", 404, "batch_images", len(batch_results))
