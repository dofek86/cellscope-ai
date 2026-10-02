"""Inference for the five locally trained V6 folds.

The predictions are research quality-control estimates, not diagnoses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.transforms import functional as TF


CELL_TYPES = ("A549", "CACO", "HPMEC", "HSAEC", "HUVEC", "NHBE")
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
GLOBAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE,
])


class V6Network(nn.Module):
    """Exact state-dict-compatible inference definition of CellScopeNet."""

    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.local_attention = nn.Sequential(
            nn.Linear(features, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.image_projection = nn.Sequential(
            nn.LayerNorm(features * 3), nn.Linear(features * 3, 384),
            nn.SiLU(), nn.Dropout(0.30),
        )
        self.image_head = nn.Linear(384, 1)
        self.context = nn.Sequential(
            nn.LayerNorm(14), nn.Linear(14, 64), nn.SiLU(), nn.Dropout(0.20)
        )
        self.context_delta = nn.Sequential(
            nn.Linear(448, 128), nn.SiLU(), nn.Dropout(0.20), nn.Linear(128, 1)
        )
        self.context_gate = nn.Sequential(nn.Linear(448, 1), nn.Sigmoid())
        self.global_head = nn.Linear(features, 1)
        self.local_head = nn.Linear(features, 1)

    def forward(self, global_view, local_views, context):
        batch, views, channels, height, width = local_views.shape
        global_features = self.backbone(global_view)
        local_features = self.backbone(
            local_views.reshape(batch * views, channels, height, width)
        ).reshape(batch, views, -1)
        attention = torch.softmax(self.local_attention(local_features).squeeze(-1), dim=1)
        pooled = torch.sum(local_features * attention.unsqueeze(-1), dim=1)
        representation = self.image_projection(torch.cat([
            global_features, pooled, torch.abs(global_features - pooled),
        ], dim=1))
        image_logit = self.image_head(representation).squeeze(1)
        fused = torch.cat([representation, self.context(context)], dim=1)
        fused_logit = image_logit + (
            self.context_gate(fused).squeeze(1)
            * self.context_delta(fused).squeeze(1)
        )
        return torch.sigmoid(image_logit), torch.sigmoid(fused_logit)


def normalize_illumination(image: Image.Image) -> Image.Image:
    """Match the exploratory flatfield transform used in the V7 pilot."""
    small = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    background = small.filter(ImageFilter.GaussianBlur(radius=48))
    background = np.asarray(
        background.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32
    )
    pixels = np.asarray(image, dtype=np.float32)
    corrected = np.clip(
        pixels * (float(background.mean()) / np.maximum(background, 16.0))[..., None],
        0, 255,
    ).astype(np.uint8)
    return Image.fromarray(corrected)


def open_microscopy_rgb(image_path: Path) -> Image.Image:
    """Read microscopy images without clipping 16-bit grayscale data to white.

    The training PNGs are ordinary 8-bit images and retain their original
    conversion. Higher-bit-depth grayscale images need an explicit display
    window before they can be passed to the 8-bit trained network.
    """
    with Image.open(image_path) as source:
        if source.mode in {"I;16", "I;16L", "I;16B", "I", "F"}:
            values = np.asarray(source, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"No finite pixels in {image_path}")
            low, high = np.percentile(finite, [1, 99])
            if high <= low:
                grayscale = np.zeros(values.shape, dtype=np.uint8)
            else:
                grayscale = np.clip((values - low) * 255 / (high - low), 0, 255)
                grayscale = np.nan_to_num(grayscale, nan=0, posinf=255, neginf=0)
                grayscale = grayscale.astype(np.uint8)
            return Image.fromarray(grayscale).convert("RGB")
        return source.convert("RGB")


def prepare_views(
    image_path: Path, local_views: int, illumination_mode: str = "raw",
    local_size: int = 128,
):
    if local_size not in {128, 160, 192, 224}:
        raise ValueError(f"Unsupported local view size: {local_size}")
    local_transform = transforms.Compose([
        transforms.Resize((local_size, local_size)), transforms.ToTensor(), NORMALIZE,
    ])
    image = open_microscopy_rgb(image_path)
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    if illumination_mode == "flatfield":
        image = normalize_illumination(image)
    elif illumination_mode != "raw":
        raise ValueError(f"Unsupported illumination mode: {illumination_mode}")
    width, height = image.size
    side = max(32, int(min(width, height) * 0.42))
    centers = [(0.50, 0.50), (0.50, 0.28), (0.50, 0.72),
               (0.35, 0.50), (0.65, 0.50)]
    patches = []
    for x, y in centers[:local_views]:
        left = min(max(0, round(x * width - side / 2)), width - side)
        top = min(max(0, round(y * height - side / 2)), height - side)
        patches.append(local_transform(TF.crop(image, top, left, side, side)))
    return GLOBAL_TRANSFORM(image).unsqueeze(0), torch.stack(patches).unsqueeze(0)


def encode_context(stats: dict, metadata: dict | None):
    metadata = metadata or {}
    defaults = stats["numeric_defaults"]
    raw = []
    for model_name, source_name in stats["numeric_context"]:
        value = metadata.get(source_name)
        missing = value is None
        raw.extend([float(missing), defaults[model_name] if missing else float(value)])
    raw.extend(float(metadata.get("cell_type") == cell) for cell in CELL_TYPES)
    result = (np.asarray(raw, dtype=np.float32) - np.asarray(stats["mean"])) / np.asarray(
        stats["scale"]
    )
    return torch.as_tensor(result, dtype=torch.float32).unsqueeze(0)


def predict(image_path: Path, checkpoint_directory: Path, metadata: dict | None = None,
            device: str | None = None) -> dict:
    """Average the five fold models on a new image; use image head if cell type is unknown."""
    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = sorted(checkpoint_directory.glob("fold_*.pt"))
    if len(paths) != 5:
        raise ValueError(f"Expected five V6 checkpoints, found {len(paths)}")
    predictions = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("view_mode") != "dual" or not 1 <= checkpoint.get("local_views", 0) <= 5:
            raise ValueError(f"Unsupported V6 view configuration in {path}")
        if (checkpoint.get("local_size", 128) not in {128, 160, 192, 224}
                or checkpoint.get("illumination_mode", "raw") != "raw"
                or abs(checkpoint.get("local_crop_fraction", 0.42) - 0.42) > 1e-8):
            raise ValueError(f"Unsupported view preprocessing in {path}")
        model = V6Network().to(chosen_device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        global_view, local_views = prepare_views(
            image_path, checkpoint["local_views"], local_size=checkpoint.get("local_size", 128)
        )
        context = encode_context(checkpoint["context_stats"], metadata)
        with torch.no_grad():
            global_view = global_view.to(chosen_device)
            local_views = local_views.to(chosen_device)
            context = context.to(chosen_device)
            image_probability, context_probability = model(
                global_view, local_views, context,
            )
            flipped_image, flipped_context = model(
                torch.flip(global_view, dims=[3]),
                torch.flip(local_views, dims=[4]), context,
            )
            image_probability = (image_probability + flipped_image) / 2
            context_probability = (context_probability + flipped_context) / 2
        predictions.append({
            "fold": int(checkpoint["fold"]),
            "image": float(image_probability.item()),
            "context": float(context_probability.item()),
        })
        del model
    if {row["fold"] for row in predictions} != set(range(5)):
        raise ValueError("Expected one checkpoint for each of the five folds")
    image_mean = float(np.mean([row["image"] for row in predictions]))
    context_mean = float(np.mean([row["context"] for row in predictions]))
    known_cell = metadata is not None and metadata.get("cell_type") in CELL_TYPES
    return {
        "probability_good": context_mean if known_cell else image_mean,
        "decision": "good" if (context_mean if known_cell else image_mean) >= 0.5 else "bad",
        "used_context": bool(known_cell),
        "image_only_probability": image_mean,
        "context_probability": context_mean,
        "fold_predictions": predictions,
        "research_only": True,
    }
