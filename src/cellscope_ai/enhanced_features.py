"""Enhanced transparent microscopy features for the strict v3 candidate."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from .features import FEATURE_NAMES, extract_features_array, gradients, load_grayscale, otsu_threshold


HISTOGRAM_NAMES = [f"intensity_hist_{index}" for index in range(16)]
ORIENTATION_NAMES = [f"gradient_orientation_{index}" for index in range(12)]
LBP_NAMES = [f"lbp_hist_{index}" for index in range(32)]
MULTISCALE_NAMES = [
    f"scale_{size}_{name}"
    for size in (64, 128)
    for name in ("std", "gradient_mean", "gradient_std", "laplacian_variance")
]
REGION_NAMES = [
    f"coarse_{kind}_{row}_{column}"
    for kind in ("foreground", "laplacian_variance")
    for row in range(4)
    for column in range(4)
]
TAIL_NAMES = [
    "intensity_below_005",
    "intensity_below_010",
    "intensity_below_020",
    "intensity_above_080",
    "intensity_above_090",
    "intensity_above_095",
]
ENHANCED_FEATURE_NAMES = (
    FEATURE_NAMES
    + HISTOGRAM_NAMES
    + ORIENTATION_NAMES
    + LBP_NAMES
    + MULTISCALE_NAMES
    + REGION_NAMES
    + TAIL_NAMES
)


def laplacian(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    result[1:-1, 1:-1] = (
        image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
        - 4 * image[1:-1, 1:-1]
    )
    return result


def lbp_histogram(image: np.ndarray) -> np.ndarray:
    center = image[1:-1, 1:-1]
    neighbors = (
        image[:-2, :-2], image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
        image[2:, 2:], image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2],
    )
    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    histogram = np.bincount((code.ravel() // 8), minlength=32).astype(np.float64)
    return histogram / max(histogram.sum(), 1.0)


def enhanced_features_array(image: np.ndarray) -> np.ndarray:
    base = extract_features_array(image)
    histogram, _ = np.histogram(image, bins=16, range=(0.0, 1.0), density=False)
    histogram = histogram.astype(np.float64) / max(histogram.sum(), 1)

    gx, gy = gradients(image)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    orientation_bins = np.minimum((orientation / np.pi * 12).astype(int), 11)
    orientation_hist = np.bincount(
        orientation_bins.ravel(), weights=magnitude.ravel(), minlength=12
    ).astype(np.float64)
    orientation_hist /= max(orientation_hist.sum(), 1e-12)

    multiscale: list[float] = []
    for size in (64, 128):
        step = image.shape[0] // size
        reduced = image.reshape(size, step, size, step).mean(axis=(1, 3))
        rgx, rgy = gradients(reduced)
        rgradient = np.hypot(rgx, rgy)
        multiscale.extend(
            [float(reduced.std()), float(rgradient.mean()), float(rgradient.std()), float(laplacian(reduced).var())]
        )

    threshold = otsu_threshold(image)
    foreground = image < threshold
    lap = laplacian(image)
    region_values: list[float] = []
    block = image.shape[0] // 4
    for source in (foreground, lap):
        for row in range(4):
            for column in range(4):
                region = source[row * block : (row + 1) * block, column * block : (column + 1) * block]
                region_values.append(float(region.mean() if source is foreground else region.var()))

    tails = [
        float((image < 0.05).mean()), float((image < 0.10).mean()), float((image < 0.20).mean()),
        float((image > 0.80).mean()), float((image > 0.90).mean()), float((image > 0.95).mean()),
    ]
    return np.concatenate(
        [base, histogram, orientation_hist, lbp_histogram(image), np.asarray(multiscale), np.asarray(region_values), np.asarray(tails)]
    )


def extract_enhanced_features(path: Path) -> np.ndarray:
    return enhanced_features_array(load_grayscale(path))


def extract_enhanced_features_file(source: BinaryIO) -> tuple[np.ndarray, Image.Image]:
    from .features import prepare_grayscale

    with Image.open(source) as image:
        preview = image.convert("RGB")
        features = enhanced_features_array(prepare_grayscale(image))
    return features, preview
