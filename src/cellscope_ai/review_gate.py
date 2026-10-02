"""Three-way research policy for a model's probability of acceptable quality.

Choose the auto-accept threshold on held-out *inner* sessions only. The outer
test labels may be used to report performance, never to choose the threshold.
"""

from __future__ import annotations

import numpy as np


THRESHOLDS = tuple(np.round(np.arange(0.50, 1.001, 0.05), 2))


def review_counts(y, probability_good, accept_threshold) -> dict[str, float | int]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(probability_good, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and scores must be nonempty matching vectors")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores must be finite")
    thresholds = np.asarray(accept_threshold, dtype=float)
    if thresholds.ndim > 1 or (thresholds.ndim == 1 and thresholds.shape != scores.shape):
        raise ValueError("Threshold must be scalar or match the scores")
    if not np.isfinite(thresholds).all() or not ((thresholds >= .5) & (thresholds <= 1)).all():
        raise ValueError("The auto-accept threshold must be between 0.5 and 1")

    auto_good = scores >= thresholds
    auto_bad = scores < 0.5
    review = ~(auto_good | auto_bad)
    bad = labels == 0
    good = ~bad
    bad_auto_good = int((bad & auto_good).sum())
    good_auto_good = int((good & auto_good).sum())
    return {
        "images": len(labels),
        "bad_images": int(bad.sum()),
        "good_images": int(good.sum()),
        "auto_good": int(auto_good.sum()),
        "auto_bad": int(auto_bad.sum()),
        "review": int(review.sum()),
        "bad_auto_good": bad_auto_good,
        "good_auto_good": good_auto_good,
        "bad_review": int((bad & review).sum()),
        "good_review": int((good & review).sum()),
        "bad_auto_bad": int((bad & auto_bad).sum()),
        "good_auto_bad": int((good & auto_bad).sum()),
        "bad_miss_fraction": bad_auto_good / int(bad.sum()) if bad.any() else float("nan"),
        "review_fraction": float(review.mean()),
        "auto_good_error_fraction": bad_auto_good / int(auto_good.sum())
        if auto_good.any() else float("nan"),
    }


def select_accept_threshold(y, probability_good, max_bad_miss_fraction: float) -> dict:
    """Choose the least restrictive preset threshold meeting an inner-session target."""
    if not 0 <= max_bad_miss_fraction < 1:
        raise ValueError("Expected a target fraction in [0, 1)")
    for threshold in THRESHOLDS:
        counts = review_counts(y, probability_good, float(threshold))
        if not counts["bad_images"]:
            raise ValueError("Inner selection set has no bad examples")
        if counts["bad_miss_fraction"] <= max_bad_miss_fraction:
            return {"threshold": float(threshold), "inner": counts,
                    "target_met_inner": True}
    return {"threshold": 1.0,
            "inner": review_counts(y, probability_good, 1.0),
            "target_met_inner": False}
