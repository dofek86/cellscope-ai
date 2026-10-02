"""Transparent image features for the first reproducible baseline.

The implementation uses only NumPy and Pillow so it also runs under restrictive
Windows application-control policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


GLOBAL_FEATURE_NAMES = [
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_p50",
    "intensity_p90",
    "entropy",
    "gradient_mean",
    "gradient_std",
    "laplacian_variance",
    "foreground_fraction",
    "foreground_row_std",
    "foreground_col_std",
    "texture_contrast_h",
    "texture_contrast_v",
    "texture_homogeneity_h",
    "texture_homogeneity_v",
]

BLOCK_GRID = 8
BLOCK_FEATURE_NAMES = [
    f"block_{kind}_{row}_{column}"
    for kind in ("mean", "std", "gradient")
    for row in range(BLOCK_GRID)
    for column in range(BLOCK_GRID)
]
FEATURE_NAMES = GLOBAL_FEATURE_NAMES + BLOCK_FEATURE_NAMES


def prepare_grayscale(image: Image.Image, size: int = 256) -> np.ndarray:
    image = image.convert("L")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), color=0)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def load_grayscale(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        return prepare_grayscale(image, size)


def otsu_threshold(image: np.ndarray) -> float:
    histogram, edges = np.histogram(image, bins=256, range=(0.0, 1.0))
    probability = histogram.astype(np.float64) / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(probability)
    weight_right = 1.0 - weight_left
    cumulative_mean = np.cumsum(probability * centers)
    mean_left = cumulative_mean / np.maximum(weight_left, 1e-12)
    mean_total = float(cumulative_mean[-1])
    mean_right = (mean_total - cumulative_mean) / np.maximum(weight_right, 1e-12)
    between = weight_left * weight_right * (mean_left - mean_right) ** 2
    return float(centers[int(np.nanargmax(between))])


def gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[1:-1, 1:-1] = (
        -image[:-2, :-2]
        - 2 * image[1:-1, :-2]
        - image[2:, :-2]
        + image[:-2, 2:]
        + 2 * image[1:-1, 2:]
        + image[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        -image[:-2, :-2]
        - 2 * image[:-2, 1:-1]
        - image[:-2, 2:]
        + image[2:, :-2]
        + 2 * image[2:, 1:-1]
        + image[2:, 2:]
    )
    return gx, gy


def extract_features_array(image: np.ndarray) -> np.ndarray:
    gx, gy = gradients(image)
    gradient = np.hypot(gx, gy)
    laplacian = np.zeros_like(image)
    laplacian[1:-1, 1:-1] = (
        image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
        - 4 * image[1:-1, 1:-1]
    )

    histogram, _ = np.histogram(image, bins=256, range=(0.0, 1.0))
    probability = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probability * np.log2(probability)).sum())

    threshold = otsu_threshold(image)
    foreground = image < threshold
    row_density = foreground.mean(axis=1)
    col_density = foreground.mean(axis=0)

    quantized = np.clip((image * 15).round(), 0, 15).astype(np.int16)
    diff_h = quantized[:, 1:] - quantized[:, :-1]
    diff_v = quantized[1:, :] - quantized[:-1, :]
    diff_h_sq = diff_h.astype(np.float64) ** 2
    diff_v_sq = diff_v.astype(np.float64) ** 2

    global_values = [
        image.mean(),
        image.std(),
        *np.quantile(image, [0.10, 0.50, 0.90]),
        entropy,
        gradient.mean(),
        gradient.std(),
        laplacian.var(),
        foreground.mean(),
        row_density.std(),
        col_density.std(),
        diff_h_sq.mean(),
        diff_v_sq.mean(),
        np.mean(1.0 / (1.0 + diff_h_sq)),
        np.mean(1.0 / (1.0 + diff_v_sq)),
    ]

    block_size = image.shape[0] // BLOCK_GRID
    block_shape = (BLOCK_GRID, block_size, BLOCK_GRID, block_size)
    block_mean = image.reshape(block_shape).mean(axis=(1, 3))
    block_std = image.reshape(block_shape).std(axis=(1, 3))
    block_gradient = gradient.reshape(block_shape).mean(axis=(1, 3))
    spatial_values = np.concatenate(
        [block_mean.ravel(), block_std.ravel(), block_gradient.ravel()]
    )
    return np.concatenate([np.asarray(global_values, dtype=np.float64), spatial_values])


def extract_features(path: Path) -> np.ndarray:
    return extract_features_array(load_grayscale(path))


def extract_features_file(source: BinaryIO) -> tuple[np.ndarray, Image.Image]:
    with Image.open(source) as image:
        preview = image.convert("RGB")
        features = extract_features_array(prepare_grayscale(image))
    return features, preview
