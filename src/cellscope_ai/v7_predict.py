"""Experimental image-only V6/flatfield ensemble for new microscopy images.

This is an unvalidated research candidate. It is not a diagnostic model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cellscope_ai.v6_predict import V6Network, prepare_views


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_GROUPS = {
    "raw": (
        PROJECT_ROOT / "artifacts/v7-pilot-baseline-folds-0",
        PROJECT_ROOT / "artifacts/v7-pilot-baseline-folds-1-2-3-4",
    ),
    "flatfield": (
        PROJECT_ROOT / "artifacts/v7-pilot-flatfield-folds-0",
        PROJECT_ROOT / "artifacts/v7-pilot-flatfield-folds-1-2-3-4",
    ),
}


def checkpoint_paths(mode: str) -> list[Path]:
    if mode not in CHECKPOINT_GROUPS:
        raise ValueError(f"Unsupported mode: {mode}")
    first, others = CHECKPOINT_GROUPS[mode]
    paths = [first / "fold_0.pt"] + [others / f"fold_{fold}.pt" for fold in range(1, 5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints: {', '.join(missing)}")
    return paths


def predict_branch(image_path: Path, mode: str, device: torch.device) -> dict:
    paths = checkpoint_paths(mode)
    global_view, local_views = prepare_views(image_path, 3, illumination_mode=mode)
    global_view = global_view.to(device)
    local_views = local_views.to(device)
    context = torch.zeros((1, 14), device=device)
    fold_probabilities = []
    for expected_fold, path in enumerate(paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("fold") != expected_fold
                or checkpoint.get("view_mode") != "dual"
                or checkpoint.get("local_views") != 3
                or checkpoint.get("illumination_mode", "raw") != mode
                or abs(checkpoint.get("local_crop_fraction", 0.42) - 0.42) > 1e-8):
            raise ValueError(f"Unexpected checkpoint configuration: {path}")
        model = V6Network().to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            ordinary = model(global_view, local_views, context)[0]
            flipped = model(
                torch.flip(global_view, dims=[3]),
                torch.flip(local_views, dims=[4]),
                context,
            )[0]
            probability = float(((ordinary + flipped) / 2).item())
        fold_probabilities.append(probability)
        del model
    return {
        "mean_probability_good": float(np.mean(fold_probabilities)),
        "fold_probabilities": fold_probabilities,
    }


def predict_candidate(image_path: Path, device: str | None = None) -> dict:
    """Use the raw and corrected five-fold models; report image-only estimates."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = predict_branch(image_path, "raw", chosen_device)
    corrected = predict_branch(image_path, "flatfield", chosen_device)
    probability = (raw["mean_probability_good"] + corrected["mean_probability_good"]) / 2
    return {
        "image": str(image_path),
        "raw_probability_good": raw["mean_probability_good"],
        "flatfield_probability_good": corrected["mean_probability_good"],
        "blend_probability_good": probability,
        "blend_decision": "good" if probability >= 0.5 else "bad",
        "raw_fold_probabilities": raw["fold_probabilities"],
        "flatfield_fold_probabilities": corrected["fold_probabilities"],
        "used_context": False,
        "research_only": True,
    }
