"""Exploratory rejection hints learned from binary expert labels.

These are associations with ``bad``, not annotated causes or explanations of
the neural network's internal decision. Keep that boundary visible in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Direction matters: the same measurement can mean different things for
# different cell types. Only show a tail when that tail is associated with
# more expert ``bad`` labels in the selected reference group.
REASON_SPECS = (
    ("focus_proxy", "low", "Malo ostrih robov", "Možna neostrina ali šibko vidna celična tekstura."),
    ("intensity_range", "low", "Nizek razpon sivin", "Celično območje ima malo kontrasta."),
    ("brightness_median", "low", "Temnejša slika", "Celično območje je temnejše od primerljivih slik."),
    ("brightness_median", "high", "Svetlejša slika", "Celično območje je svetlejše od primerljivih slik."),
    ("dark_fraction", "high", "Več zelo temnih pikslov", "Del območja je izrazito temen."),
    ("bright_fraction", "high", "Več zelo svetlih pikslov", "Del območja je zelo svetel; podrobnosti se lahko izgubijo."),
    ("illumination_spread", "high", "Neenakomerna svetlost", "Svetlost se med deli slike precej spreminja."),
    ("texture_proxy", "low", "Malo drobne teksture", "Drobne strukture so manj izrazite kot na primerljivih slikah."),
    ("texture_spread", "high", "Neenakomerna tekstura", "Tekstura se med deli slike precej spreminja."),
)


def suggest_rejection_factors(
    values: dict[str, float], reference: pd.DataFrame, max_items: int = 3,
) -> list[dict[str, object]]:
    """Return supported, image-specific associations with expert ``bad`` labels.

    The reference should already be restricted to the known cell type when
    possible. This diagnostic is deliberately conservative because sessions
    are correlated and the source only provides a binary label.
    """
    if max_items < 1 or len(reference) < 100 or "label" not in reference:
        return []
    labels = pd.to_numeric(reference["label"], errors="coerce").to_numpy()
    valid = np.isin(labels, [0, 1])
    if valid.sum() < 100 or min((labels[valid] == 0).sum(), (labels[valid] == 1).sum()) < 20:
        return []
    sessions = reference["image_id"].astype(str).str.split("_").str[0].to_numpy()
    candidates = []
    for feature, direction, title, detail in REASON_SPECS:
        if feature not in values or feature not in reference:
            continue
        observed = float(values[feature])
        distribution = pd.to_numeric(reference[feature], errors="coerce").to_numpy(float)
        usable = valid & np.isfinite(distribution)
        if not np.isfinite(observed) or usable.sum() < 100:
            continue
        cut = float(np.quantile(distribution[usable], 0.20 if direction == "low" else 0.80))
        tail = usable & (distribution <= cut if direction == "low" else distribution >= cut)
        outside = usable & ~tail
        if not (observed <= cut if direction == "low" else observed >= cut):
            continue
        if tail.sum() < 25 or outside.sum() < 50:
            continue
        # A few neighboring images from one acquisition are not independent
        # confirmation of a visible pattern.
        if len(np.unique(sessions[tail])) < 3 or len(np.unique(sessions[tail & (labels == 0)])) < 2:
            continue
        bad_tail = float(np.mean(labels[tail] == 0))
        bad_outside = float(np.mean(labels[outside] == 0))
        lift = bad_tail - bad_outside
        if lift < 0.12:
            continue
        candidates.append({
            "feature": feature,
            "direction": direction,
            "title": title,
            "detail": detail,
            "bad_rate_tail": round(bad_tail, 3),
            "bad_rate_other": round(bad_outside, 3),
            "reference_images": int(usable.sum()),
            "tail_images": int(tail.sum()),
            "risk_difference": round(lift, 3),
        })
    candidates.sort(key=lambda item: item["risk_difference"], reverse=True)
    return candidates[:max_items]


def describe_visible_outliers(
    values: dict[str, float], reference: pd.DataFrame, max_items: int = 2,
) -> list[dict[str, object]]:
    """Describe unusual pixels when binary labels cannot support a reason.

    These observations do not imply an association with the model's decision.
    """
    if max_items < 1 or len(reference) < 25:
        return []
    observations = []
    for feature, direction, title, detail in REASON_SPECS:
        if feature not in values or feature not in reference:
            continue
        observed = float(values[feature])
        distribution = pd.to_numeric(reference[feature], errors="coerce").to_numpy(float)
        distribution = distribution[np.isfinite(distribution)]
        if not np.isfinite(observed) or len(distribution) < 25:
            continue
        percentile = float(
            np.mean(distribution < observed) + 0.5 * np.mean(distribution == observed)
        )
        if direction == "low" and percentile > 0.15:
            continue
        if direction == "high" and percentile < 0.85:
            continue
        observations.append({
            "feature": feature,
            "title": title,
            "detail": detail,
            "percentile": round(percentile * 100),
            "reference_images": len(distribution),
            "extremeness": abs(percentile - 0.5),
        })
    observations.sort(key=lambda item: item["extremeness"], reverse=True)
    return observations[:max_items]
