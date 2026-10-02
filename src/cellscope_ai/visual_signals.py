"""Uncalibrated, interpretable image measurements for research error review.

These describe the pixels, not cell viability or the reason for an expert label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


SIGNAL_NAMES = (
    "brightness_median",
    "intensity_range",
    "dark_fraction",
    "bright_fraction",
    "focus_proxy",
    "texture_proxy",
    "illumination_spread",
    "texture_spread",
)


def measure_array(gray: np.ndarray) -> dict[str, float]:
    """Measure a grayscale array in [0, 1] without guessing cell boundaries."""
    if gray.ndim != 2 or min(gray.shape) < 32 or not np.isfinite(gray).all():
        raise ValueError("Expected a finite grayscale image of at least 32 x 32 pixels")
    if gray.min() < 0 or gray.max() > 1:
        raise ValueError("Grayscale values must be in [0, 1]")

    # The central window reduces the effect of the frame, while still covering
    # the horizontal and vertical channel examples in the current collection.
    height, width = gray.shape
    roi = gray[height // 8 : height - height // 8, width // 5 : width - width // 5]
    lap = (
        roi[:-2, 1:-1] + roi[2:, 1:-1]
        + roi[1:-1, :-2] + roi[1:-1, 2:]
        - 4 * roi[1:-1, 1:-1]
    )
    dx = np.diff(roi, axis=1)
    dy = np.diff(roi, axis=0)
    tile_medians, tile_textures = [], []
    for row in range(4):
        for column in range(4):
            tile = roi[
                row * roi.shape[0] // 4 : (row + 1) * roi.shape[0] // 4,
                column * roi.shape[1] // 4 : (column + 1) * roi.shape[1] // 4,
            ]
            tile_medians.append(float(np.median(tile)))
            tile_textures.append(float(np.std(tile)))
    p05, p95 = np.quantile(roi, [0.05, 0.95])
    return {
        "brightness_median": float(np.median(roi)),
        "intensity_range": float(p95 - p05),
        "dark_fraction": float((roi < 0.04).mean()),
        "bright_fraction": float((roi > 0.96).mean()),
        "focus_proxy": float(np.var(lap)),
        "texture_proxy": float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2),
        "illumination_spread": float(np.std(tile_medians)),
        "texture_spread": float(np.std(tile_textures)),
    }


def measure_image(path: Path, size: tuple[int, int] = (512, 384)) -> dict[str, float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize(size, Image.Resampling.LANCZOS),
                          dtype=np.float32) / 255.0
    return measure_array(gray)
