"""Inference for the full-data CellScope DINOv2 research candidate."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from .dino_head import predict_bundle
from .relative_texture_review import CELL_TYPES, MIN_BATCH_SIZE, context_omission_stability
from .visual_signals import measure_image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts/dino-candidate-deployment/full_data.joblib"
TEXTURE_REVIEW_PATH = ROOT / "artifacts/dino-candidate-deployment/relative_texture_review.joblib"
VIEW_CENTERS = ((.50, .50), (.50, .28), (.50, .72), (.35, .50), (.65, .50))
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def channel_proxy(image: Image.Image) -> Image.Image:
    """Exact geometry used in scripts/channel_proxy.py, without importing scripts."""
    width, height = image.size
    preview = np.asarray(image.convert("L").resize((128, 96), Image.Resampling.BILINEAR),
                         dtype=np.float32)
    center = np.median(preview[24:72, 32:96])
    horizontal = np.median(np.concatenate((preview[:8].ravel(), preview[-8:].ravel())))
    vertical = np.median(np.concatenate((preview[:, :10].ravel(), preview[:, -10:].ravel())))
    if center - horizontal >= center - vertical:
        xmargin, ymargin = .05, .14
    else:
        xmargin, ymargin = .14, .05
    return image.crop((round(xmargin * width), round(ymargin * height),
                       round((1 - xmargin) * width), round((1 - ymargin) * height)))


def prepare_views(path: Path, size: int, proxy: bool = False) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    global_image = channel_proxy(image) if proxy else image
    width, height = image.size
    side = max(32, int(min(width, height) * .42))
    views = [global_image.resize((size, size), Image.Resampling.BILINEAR)]
    for cx, cy in VIEW_CENTERS:
        left = min(max(0, round(cx * width - side / 2)), width - side)
        top = min(max(0, round(cy * height - side / 2)), height - side)
        crop = image.crop((left, top, left + side, top + side))
        views.append(crop.resize((size, size), Image.Resampling.BILINEAR))
    return torch.stack([torch.from_numpy(((np.asarray(view, dtype=np.float32) / 255
                                          - MEAN) / STD).transpose(2, 0, 1).copy())
                        for view in views])


class DinoCandidate:
    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bundle = joblib.load(model_path)
        self.texture_review = (joblib.load(TEXTURE_REVIEW_PATH)
                               if TEXTURE_REVIEW_PATH.exists() else None)
        self.small = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    trust_repo=True).eval().to(self.device)
        self.base = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                                   trust_repo=True).eval().to(self.device)

    def _encode(self, model, views: torch.Tensor) -> np.ndarray:
        context = (torch.autocast("cuda", dtype=torch.float16)
                   if self.device.type == "cuda" else nullcontext())
        with torch.inference_mode(), context:
            output = model.forward_features(views.to(self.device))
            cls = output["x_norm_clstoken"]
            mean_patch = output["x_norm_patchtokens"].mean(dim=1)
            return torch.cat([cls, mean_patch], dim=1).float().cpu().numpy()

    def features_for(self, path: Path) -> dict[str, np.ndarray]:
        small = self._encode(self.small, prepare_views(path, 224))
        base = self._encode(self.base, prepare_views(path, 224))
        large_view = self._encode(self.base, prepare_views(path, 336))
        # The proxy changes only the global view; five local patches are identical.
        proxy_global = self._encode(self.base, prepare_views(path, 224, proxy=True)[:1])
        proxy = base.copy()
        proxy[0] = proxy_global[0]
        return {"small_224": small, "base_224": base,
                "base_336": large_view, "channel_proxy": proxy}

    def predict_paths(self, paths: list[Path], session_ids: list[str] | None = None,
                      cell_type: str | None = None) -> list[dict]:
        if not paths:
            return []
        if session_ids is None:
            session_ids = [str(i) for i in range(len(paths))]
        if len(paths) != len(session_ids):
            raise ValueError("Each image needs one session ID")
        if cell_type is not None and cell_type not in CELL_TYPES:
            raise ValueError("Unknown cell type for texture review")
        features = [self.features_for(path) for path in paths]
        stacked = {name: np.stack([item[name] for item in features])
                   for name in features[0]}
        result = predict_bundle(self.bundle, stacked, np.asarray(session_ids, dtype=str))
        texture_flag = None
        texture_stability = None
        if (cell_type is not None and self.texture_review is not None
                and len(paths) >= MIN_BATCH_SIZE and len(set(session_ids)) == 1):
            texture = np.asarray([measure_image(path)["texture_proxy"] for path in paths])
            head = self.texture_review
            def auxiliary_predict(matrix: np.ndarray) -> np.ndarray:
                return head["model"].predict_proba(
                    head["scaler"].transform(matrix))[:, 1]

            def primary_for_subset(included: np.ndarray) -> np.ndarray:
                subset = {name: values[included] for name, values in stacked.items()}
                return predict_bundle(self.bundle, subset,
                                      np.asarray(session_ids, dtype=str)[included])["blend"]

            texture_flag, texture_stability = context_omission_stability(
                texture, cell_type, result["blend"], auxiliary_predict,
                primary_for_subset)
        return [{"image": path.name, "session": session_ids[i],
                 "probability_good": float(result["blend"][i]),
                 "decision": "good" if result["blend"][i] >= .5 else "bad",
                 "base_probability_good": float(result["base"][i]),
                 "patch_probability_good": float(result["patch"][i]),
                 "texture_review_flag": (bool(texture_flag[i]) if texture_flag is not None
                                         else None),
                 "texture_review_stability": (float(texture_stability[i])
                                              if texture_stability is not None else None)}
                for i, path in enumerate(paths)]
