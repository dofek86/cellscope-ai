"""Dataset discovery and metadata loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
LABELS = {"bad": 0, "good": 1}
SPLITS = {"train", "val", "validation", "test"}


def _path_token(path: Path, choices: set[str]) -> str | None:
    for part in (value.lower() for value in path.parts):
        if part in choices:
            return part
    return None


def discover_images(root: Path) -> pd.DataFrame:
    """Build an image manifest from the extracted Zenodo directory tree."""

    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_name = _path_token(path.relative_to(root), set(LABELS))
        split = _path_token(path.relative_to(root), SPLITS)
        if split == "validation":
            split = "val"
        records.append(
            {
                "image_id": path.stem,
                "image_path": str(path.resolve()),
                "label_name": label_name,
                "label": LABELS.get(label_name) if label_name else None,
                "split": split,
            }
        )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise FileNotFoundError(f"No supported images found under {root}")
    return manifest


def load_metadata(workbook: Path) -> pd.DataFrame:
    """Load the original metadata without changing the source workbook."""

    metadata = pd.read_excel(workbook, sheet_name="Main")
    return metadata.rename(
        columns={
            "imageID": "image_id",
            "cell type": "cell_type",
            "seeding density, cells/ml": "seeding_density_cells_ml",
            "time after seeding, h": "time_after_seeding_h",
            "Decision 1/2 (good/bad)": "source_label",
            "flow rate": "flow_rate",
        }
    )


def build_manifest(image_root: Path, workbook: Path) -> pd.DataFrame:
    """Join image paths to source metadata and validate label consistency."""

    images = discover_images(image_root)
    metadata = load_metadata(workbook)
    manifest = images.merge(metadata, on="image_id", how="left", validate="one_to_one")

    missing_metadata = int(manifest["cell_type"].isna().sum())
    if missing_metadata:
        raise ValueError(f"Metadata missing for {missing_metadata} discovered images")

    # The source column describes 1/2 as good/bad. Keep directory names as the
    # training target and use this check to catch archive/layout misunderstandings.
    expected_source_label = manifest["label_name"].map({"good": 1, "bad": 2})
    inconsistent = manifest[expected_source_label != manifest["source_label"]]
    if not inconsistent.empty:
        raise ValueError(f"Found {len(inconsistent)} inconsistent labels")
    return manifest
