"""Minimal inference for the saved CellScope DINOv2 shallow head.

This module deliberately has no dependency on training scripts or datasets,
so a reviewer can run the packaged model from a clean checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def _view(raw: np.ndarray, name: str) -> np.ndarray:
    global_embedding = raw[:, 0]
    local = raw[:, 1:]
    if name == "global":
        return global_embedding
    if name == "global_mean":
        return np.concatenate((global_embedding, local.mean(axis=1)), axis=1)
    if name == "global_mean_std":
        return np.concatenate((global_embedding, local.mean(axis=1),
                               local.std(axis=1)), axis=1)
    raise ValueError(f"Unknown saved global view: {name}")


def _global_probability(head: dict, raw: np.ndarray) -> np.ndarray:
    matrix = _view(raw, head["view"])
    transformed = head["pca"].transform(head["scaler"].transform(matrix))
    if head["kind"].startswith(("lr_", "knn_", "hgb_")):
        return head["model"].predict_proba(transformed)[:, 1]
    return expit(head["model"].decision_function(transformed))


def _patch_probability(head: dict, raw: np.ndarray) -> np.ndarray:
    patches = raw[:, 1:]
    n_images, n_patches, n_features = patches.shape
    matrix = patches.reshape(-1, n_features)
    transformed = head["pca"].transform(head["scaler"].transform(matrix))
    p = head["model"].predict_proba(transformed)[:, 1].reshape(n_images, n_patches)
    if head["pool"] == "mean":
        return p.mean(axis=1)
    if head["pool"] == "minimum":
        return p.min(axis=1)
    if head["pool"] == "midpoint":
        return .5 * (p.mean(axis=1) + p.min(axis=1))
    raise ValueError(f"Unknown saved patch pooling: {head['pool']}")


def predict_bundle(bundle: dict, inputs: dict[str, np.ndarray],
                   sessions: np.ndarray) -> dict[str, np.ndarray]:
    if not inputs or any(len(raw) != len(sessions) for raw in inputs.values()):
        raise ValueError("Input features and session IDs differ")
    global_probabilities = [_global_probability(head, inputs[name])
                            for name, head in bundle["global_heads"].items()]
    base_raw = np.mean(global_probabilities, axis=0)
    alpha = float(bundle["alpha"])
    if alpha == 0:
        base = base_raw.copy()
    else:
        session_mean = pd.Series(base_raw).groupby(sessions).transform("mean").to_numpy()
        base = (1 - alpha) * base_raw + alpha * session_mean
    patch = np.mean([_patch_probability(head, inputs[name])
                     for name, head in bundle["patch_heads"].items()], axis=0)
    return {"base_raw": base_raw, "base": base, "patch": patch,
            "blend": .8 * base + .2 * patch}
