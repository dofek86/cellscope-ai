"""Recompute reported development metrics from packaged held-out predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"original": (0.810281490346842, 251),
            "newfolds": (0.7981285638331077, 265)}


def main() -> None:
    for split, (expected_ba, expected_bad_as_good) in EXPECTED.items():
        path = ROOT / "results" / f"{split}-oof-predictions.csv"
        frame = pd.read_csv(path, dtype={"image_id": str, "session": str})
        if len(frame) != 3072 or frame.image_id.duplicated().any():
            raise ValueError(f"Invalid cohort in {path}")
        y = frame.label.to_numpy(int)
        p = frame.blend_probability_good.to_numpy(float)
        if not set(np.unique(y)) == {0, 1} or not np.isfinite(p).all():
            raise ValueError(f"Invalid labels or probabilities in {path}")
        ba = float(balanced_accuracy_score(y, p >= 0.5))
        auc = float(roc_auc_score(y, p))
        bad_as_good = int(np.sum((y == 0) & (p >= 0.5)))
        if abs(ba - expected_ba) > 1e-10 or bad_as_good != expected_bad_as_good:
            raise ValueError(f"Unexpected {split} result")
        print(f"{split}: {len(frame)} images, {frame.session.nunique()} sessions, "
              f"BA={ba:.4%}, ROC-AUC={auc:.4f}, bad->good={bad_as_good}")


if __name__ == "__main__":
    main()
