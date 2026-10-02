# CellScope AI — public reviewer code package

This package runs the frozen DINOv2 research candidate and the public demo
interface available at https://cellscope.mitja.si/. It contains our
shallow-model head, source code, held-out development predictions, and the
eight-page technical report in `docs/CellScope-AI-Technical-Report.pdf`. It does
not contain source microscopy images, private review-queue data, or pretrained
DINOv2 backbone weights. The latter are downloaded from Meta's public
repository on first run. An internet connection is therefore needed once.

## Requirements and first run

Python 3.12, enough RAM for DINOv2-S/B, and internet access for the initial
backbone download. A GPU is optional; the smoke test uses CPU. From the root
of this extracted package, in PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv\Scripts\python.exe scripts\verify_reviewer_metrics.py
& .\.venv\Scripts\python.exe scripts\smoke_reviewer_release.py
& .\.venv\Scripts\python.exe scripts\run_dino_demo.py
```

Then open <http://127.0.0.1:7862/>. Upload one image, or up to 20 images from
**one acquisition session**. For the optional experimental relative-texture
warning, enter a session ID and cell type and upload at least five images.
Without that information, the main image model still runs. The optional
`/review` page creates local files under `data/review_queue`; nothing is
uploaded to an external service by this package. The app binds only to
127.0.0.1.

To run the same read-only interface as the public site instead, stop the local
app and run `scripts/run_public_demo.py`. Open <http://127.0.0.1:7863/>. This
version accepts one to five images, does not expose `/review`, does not retain
uploads, and listens only on loopback. Do not expose it directly to the
Internet without a TLS reverse proxy and request-size/rate limits. The HTML
and original vector logo are included in the package.

For a quick source-level check without downloading the large DINOv2 backbones,
install `pytest` and run `python -m pytest tests/test_public_demo.py -q`.

The synthetic smoke image tests code execution only. Its label and
probability have no biological interpretation. `verify_reviewer_metrics.py`
recomputes scores from saved out-of-fold predictions; it does **not** retrain
the backbones or replace a new-session validation study.

## Method and limits

The model averages four global DINOv2 representations (S/14 at 224 pixels,
B/14 at 224 and 336 pixels, and a conservative channel crop with B/14) and
three weakly supervised local-patch representations. Global and local scores
have fixed weights 80% and 20%. Multiple images from one acquisition session
permit within-session smoothing; a single image gets no such gain.

The saved development scores are 81.03% and 79.81% balanced accuracy on two
session-grouped partitions of the **same** 3,072 images. There are 251 and
265 source-labelled bad images incorrectly called good. Architecture choices
were informed by this cohort; these are exploratory estimates. The included
full-data head has **no independent score on new acquisition sessions**.
This is research decision support, not disease diagnosis or an autonomous
laboratory verdict. The model cannot give a verified biological reason for an
individual rejection.

## Sources and rights

- OOC image source: https://zenodo.org/records/10203721 . The dataset paper
  lists CC BY-SA and should be cited:
  https://www.mdpi.com/2306-5729/9/2/28 . Download raw images from the
  original record if you need to reproduce training. No original images are
  redistributed here.
- Meta DINOv2 repository and standard backbone weights:
  https://github.com/facebookresearch/dinov2 . Its repository describes the
  code and weights as Apache 2.0. First-run `torch.hub` access uses that
  source. Verify the exact upstream version before further redistribution.
- The model head is a Python `joblib` object. Load it only from a trusted
  package after checking `MANIFEST.sha256`.

## What this package does not reproduce yet

The provided per-image CSVs allow exact recomputation of the reported
development scores. A full end-to-end *retraining* replication additionally
requires the 6.7 GB source dataset, GPU feature extraction, all model-selection
logs, and a frozen environment. These research materials are documented in
the main project but are not all part of this compact inference package.
The package should therefore be described as a runnable inference and metric
replay artifact, not as complete independent model-training reproduction.

## Suggested reviewer sequence

1. Inspect `docs/CellScope-AI-Technical-Report.pdf` and the source under
   `src/cellscope_ai/`.
2. Run `scripts/verify_reviewer_metrics.py` to recompute development scores
   from saved out-of-fold predictions in `results/`.
3. Run `scripts/smoke_reviewer_release.py` for one CPU inference on a synthetic
   image, or upload an authorized microscopy image to the local demo.
4. Compare the interactive public site with this package. The public site
   serves a full-data refit and is **not** an independent validation set.
