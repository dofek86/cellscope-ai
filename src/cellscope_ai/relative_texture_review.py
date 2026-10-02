"""Research-only, batch-relative texture review signal.

The signal is an optional request for human review, never a new quality label.
It requires multiple images from the same acquisition session and cell type.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

CELL_TYPES = ("A549", "CACO", "HPMEC", "HSAEC", "HUVEC", "NHBE")
REVIEW_WEIGHT = .10
MIN_BATCH_SIZE = 5


def feature_matrix(texture: np.ndarray, cell_types: list[str],
                   batches: list[str]) -> np.ndarray:
    values = np.asarray(texture, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite, non-negative texture measurements")
    if len(values) != len(cell_types) or len(values) != len(batches):
        raise ValueError("Each image needs a cell type and a batch identity")
    if any(cell not in CELL_TYPES for cell in cell_types):
        raise ValueError("Unknown cell type for relative texture")
    log_texture = np.log(np.maximum(values, 1e-8))
    group_values: dict[tuple[str, str], list[float]] = {}
    for value, cell, batch in zip(log_texture, cell_types, batches, strict=True):
        group_values.setdefault((batch, cell), []).append(float(value))
    medians = {key: float(np.median(group)) for key, group in group_values.items()}
    deviation = np.asarray([
        value - medians[(batch, cell)]
        for value, cell, batch in zip(log_texture, cell_types, batches, strict=True)
    ])
    base = np.column_stack((log_texture, deviation, np.abs(deviation)))
    indicators = np.column_stack([
        np.asarray([cell == known for cell in cell_types], dtype=float)
        for known in CELL_TYPES
    ])
    return np.column_stack((base, indicators,
                            *(base[:, i, None] * indicators for i in range(3))))


def review_flags(probability_good: np.ndarray, auxiliary_probability_good: np.ndarray,
                 batch_size: int) -> np.ndarray:
    base = np.asarray(probability_good, dtype=float)
    aux = np.asarray(auxiliary_probability_good, dtype=float)
    if base.shape != aux.shape or base.ndim != 1:
        raise ValueError("Main and auxiliary probabilities must align")
    if not np.isfinite(base).all() or not np.isfinite(aux).all():
        raise ValueError("Probabilities must be finite")
    if batch_size < MIN_BATCH_SIZE:
        return np.zeros(len(base), dtype=bool)
    adjusted = (1 - REVIEW_WEIGHT) * base + REVIEW_WEIGHT * aux
    return (base >= .5) & (adjusted < .5)


def context_omission_stability(
    texture: np.ndarray, cell_type: str, probability_good: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    primary_for_subset: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Check whether dropping one *other* image changes each review flag.

    This measures local context sensitivity, not biological correctness. With
    five images a drop would leave fewer than the validated minimum, so no
    stability value is reported.
    """
    values = np.asarray(texture, dtype=float)
    base = np.asarray(probability_good, dtype=float)
    if values.ndim != 1 or base.ndim != 1 or len(values) != len(base):
        raise ValueError("Texture and primary probabilities must align")
    if cell_type not in CELL_TYPES:
        raise ValueError("Unknown cell type for relative texture")
    count = len(values)
    if count < MIN_BATCH_SIZE:
        raise ValueError("At least five images are needed for a texture review")
    full = feature_matrix(values, [cell_type] * count, ["batch"] * count)
    initial = review_flags(base, np.asarray(predictor(full), dtype=float), count)
    if count == MIN_BATCH_SIZE:
        return initial, None
    agreement = np.zeros(count, dtype=int)
    observations = np.zeros(count, dtype=int)
    for omitted in range(count):
        included = np.arange(count) != omitted
        subset_base = (np.asarray(primary_for_subset(included), dtype=float)
                       if primary_for_subset is not None else base[included])
        matrix = feature_matrix(values[included], [cell_type] * (count - 1),
                                ["batch"] * (count - 1))
        changed = review_flags(subset_base, np.asarray(predictor(matrix), dtype=float),
                               count - 1)
        agreement[included] += changed == initial[included]
        observations[included] += 1
    return initial, agreement / observations
