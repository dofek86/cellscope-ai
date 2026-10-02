"""Data-free end-to-end inference check for the reviewer release.

The generated image is synthetic and has no biological meaning. This checks
that public DINOv2 backbones, our saved head, and preprocessing run together.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from cellscope_ai.dino_predict import DinoCandidate


def main() -> None:
    rng = np.random.default_rng(20260925)
    height, width = 256, 320
    yy, xx = np.mgrid[:height, :width]
    texture = rng.normal(0, 7, size=(height, width))
    field = 123 + 15 * np.sin(xx / 15) + 8 * np.cos(yy / 22) + texture
    image = np.clip(field, 0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic-smoke.png"
        Image.fromarray(image).convert("RGB").save(path)
        model = DinoCandidate(device="cpu")
        results = model.predict_paths([path], session_ids=["smoke-session"])
    if len(results) != 1:
        raise ValueError("Expected one prediction")
    result = results[0]
    probability = float(result["probability_good"])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid output probability")
    if result["decision"] not in ("good", "bad"):
        raise ValueError("Invalid output decision")
    print(f"PASS: one synthetic image; P(good)={probability:.4f}; "
          f"decision={result['decision']}")


if __name__ == "__main__":
    main()
