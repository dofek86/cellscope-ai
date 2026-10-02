"""Shared encoding for experimental context fields."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


CELL_TYPES = ("A549", "CACO", "HPMEC", "HSAEC", "HUVEC", "NHBE")
NUMERIC_CONTEXT = (
    ("day_context", "day"),
    ("seeding_density_context", "seeding_density_cells_ml"),
    ("time_after_seeding_context", "time_after_seeding_h"),
    ("flow_rate_context", "flow_rate"),
)
CONTEXT_FEATURE_NAMES = [
    feature
    for model_name, _ in NUMERIC_CONTEXT
    for feature in (f"{model_name}_missing", model_name)
] + [f"cell_type_{cell_type}" for cell_type in CELL_TYPES]


def parse_number(value: object) -> float | None:
    """Extract a decimal number from dataset or form text."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def encode_context(record: Mapping[str, object], defaults: Mapping[str, float]) -> np.ndarray:
    """Encode one context record with explicit missing-value flags."""
    values: list[float] = []
    for model_name, source_name in NUMERIC_CONTEXT:
        parsed = parse_number(record.get(source_name))
        values.extend(
            [
                1.0 if parsed is None else 0.0,
                float(defaults[model_name]) if parsed is None else parsed,
            ]
        )

    selected = str(record.get("cell_type", "")).upper().strip()
    values.extend(1.0 if selected == cell_type else 0.0 for cell_type in CELL_TYPES)
    return np.asarray(values, dtype=np.float64)
