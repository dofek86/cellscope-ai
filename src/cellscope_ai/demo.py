"""Local CellScope AI v3 demo with image and experimental context."""

from __future__ import annotations

import base64
import cgi
import csv
import html
import io
import json
import zipfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .context import CELL_TYPES, CONTEXT_FEATURE_NAMES, encode_context
from .enhanced_features import ENHANCED_FEATURE_NAMES, extract_enhanced_features_file
from .features import BLOCK_GRID, FEATURE_NAMES
from .model import LogisticBaseline


ARTIFACTS = Path("artifacts/enhanced")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 200 * 1024 * 1024
MAX_BATCH_IMAGES = 500


def data_url(image: Image.Image, *, quality: int = 88) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def explanation_image(image: Image.Image, contributions: np.ndarray) -> Image.Image:
    prepared = image.convert("RGB")
    prepared.thumbnail((768, 768), Image.Resampling.LANCZOS)
    block_values = contributions[16:].reshape(3, BLOCK_GRID, BLOCK_GRID).sum(axis=0)
    scale = max(float(np.abs(block_values).max()), 1e-9)
    normalized = block_values / scale
    overlay = np.stack(
        [
            np.clip(normalized, 0.0, 1.0) * 255,
            np.zeros_like(normalized),
            np.clip(-normalized, 0.0, 1.0) * 255,
            np.abs(normalized) * 150,
        ],
        axis=-1,
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGBA").resize(
        prepared.size, Image.Resampling.BILINEAR
    )
    result = prepared.convert("RGBA")
    result.alpha_composite(overlay_image)
    return result.convert("RGB")


def pretty_name(name: str) -> str:
    replacements = {
        "day context": "culture day",
        "seeding density context": "seeding density",
        "time after seeding context": "hours after seeding",
        "flow rate context": "flow rate",
        "cell type": "cell type",
    }
    clean = name.replace("_missing", " missing").replace("_", " ")
    for source, target in replacements.items():
        clean = clean.replace(source, target)
    return clean.capitalize()


def rank_signals(names: list[str], values: np.ndarray, count: int = 6) -> list[tuple[str, float]]:
    ranked = np.argsort(np.abs(values))[::-1][:count]
    return [(pretty_name(names[index]), float(values[index])) for index in ranked]


def analyze(
    upload: io.BytesIO,
    form_context: dict[str, object],
    image_model: LogisticBaseline,
    context_model: LogisticBaseline,
    config: dict[str, Any],
    *,
    filename: str = "image",
    include_visuals: bool = True,
) -> dict[str, object]:
    image_features, preview = extract_enhanced_features_file(upload)
    context_features = encode_context(form_context, config["context_defaults"])
    combined = np.concatenate([image_features, context_features])

    image_probability = float(image_model.predict_proba(image_features.reshape(1, -1))[0])
    context_probability = float(context_model.predict_proba(combined.reshape(1, -1))[0])
    known_cell_type = str(form_context.get("cell_type", "")).upper() in CELL_TYPES
    mode = "Image + experimental context" if known_cell_type else "Image-only fallback"
    probability = context_probability if known_cell_type else image_probability
    threshold_key = "image_plus_context" if known_cell_type else "image_only"
    threshold = float(config["decision_thresholds"][threshold_key])
    review_margin = float(
        config.get("review_policy", {}).get("margins", {}).get(threshold_key, 0.15)
    )
    contributions = context_model.contributions(combined)
    image_contributions = contributions[: len(ENHANCED_FEATURE_NAMES)]
    context_contributions = contributions[len(ENHANCED_FEATURE_NAMES) :]
    return {
        "image_probability": image_probability,
        "context_probability": context_probability,
        "final_probability": probability,
        "filename": filename,
        "decision": "Acceptable culture quality" if probability >= threshold else "Review recommended",
        "assessment_mode": mode,
        "uncertain": abs(probability - threshold) < review_margin,
        "threshold": threshold,
        "review_margin": review_margin,
        "preview": data_url(preview) if include_visuals else "",
        "explanation": data_url(explanation_image(preview, image_contributions[: len(FEATURE_NAMES)])) if include_visuals else "",
        "image_signals": rank_signals(ENHANCED_FEATURE_NAMES, image_contributions),
        "context_signals": rank_signals(CONTEXT_FEATURE_NAMES, context_contributions, 5),
    }


def signal_list(signals: object) -> str:
    return "".join(
        f"<li><span>{html.escape(name)}</span><strong class={'positive' if value >= 0 else 'negative'}>{value:+.3f}</strong></li>"
        for name, value in signals  # type: ignore[union-attr]
    )


def score_card(title: str, probability: float, *, primary: bool = False) -> str:
    return f"""<article class="score-card{' primary' if primary else ''}">
      <small>{html.escape(title)}</small><strong>{probability * 100:.1f}%</strong>
      <div class="meter"><i style="width:{probability * 100:.1f}%"></i></div>
      <span>probability of acceptable quality</span></article>"""


def build_csv_report(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=(
            "filename", "decision", "assessment_mode", "image_probability",
            "context_probability", "final_probability", "uncertain", "threshold",
            "review_margin",
        ),
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in writer.fieldnames})
    return output.getvalue().encode("utf-8-sig")


def build_pdf_report(rows: list[dict[str, object]]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=landscape(A4), rightMargin=28, leftMargin=28)
    styles = getSampleStyleSheet()
    acceptable = sum(row["decision"] == "Acceptable culture quality" for row in rows)
    uncertain = sum(bool(row["uncertain"]) for row in rows)
    content: list[object] = [
        Paragraph("CellScope AI — batch quality report", styles["Title"]),
        Paragraph(datetime.now(UTC).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]),
        Spacer(1, 10),
        Paragraph(
            f"Images: {len(rows)} &nbsp;&nbsp; Acceptable: {acceptable} &nbsp;&nbsp; "
            f"Review recommended: {len(rows) - acceptable} &nbsp;&nbsp; Uncertain: {uncertain}",
            styles["Heading2"],
        ),
        Spacer(1, 10),
    ]
    data = [["Image", "Decision", "Mode", "Image", "Context", "Final", "Uncertain"]]
    for row in rows:
        data.append(
            [
                str(row["filename"])[:42], str(row["decision"]), str(row["assessment_mode"]),
                f"{float(row['image_probability']) * 100:.1f}%",
                f"{float(row['context_probability']) * 100:.1f}%",
                f"{float(row['final_probability']) * 100:.1f}%",
                "yes" if row["uncertain"] else "no",
            ]
        )
    table = Table(data, repeatRows=1, colWidths=[160, 120, 105, 55, 55, 55, 55])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c9b63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cad8d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7f4")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    content.extend([table, Spacer(1, 10), Paragraph(
        "Research quality-control signal only. Predictions require expert validation and are not clinical advice.",
        styles["Normal"],
    )])
    document.build(content)
    return output.getvalue()


def uploaded_images(items: list[Any]) -> tuple[list[tuple[str, bytes]], bool]:
    images: list[tuple[str, bytes]] = []
    used_archive = False
    for item in items:
        filename = Path(item.filename or "image").name
        payload = item.file.read(MAX_REQUEST_BYTES + 1)
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError(f"{filename} exceeds the request limit.")
        if filename.lower().endswith(".zip"):
            used_archive = True
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    suffix = Path(member.filename).suffix.lower()
                    if member.is_dir() or suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                        continue
                    if member.file_size > MAX_IMAGE_BYTES:
                        raise ValueError(f"Image {member.filename} exceeds 20 MB.")
                    images.append((Path(member.filename).name, archive.read(member)))
        else:
            if len(payload) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image {filename} exceeds 20 MB.")
            images.append((filename, payload))
        if len(images) > MAX_BATCH_IMAGES:
            raise ValueError(f"A batch may contain at most {MAX_BATCH_IMAGES} images.")
    if not images:
        raise ValueError("No supported microscopy images were found.")
    return images, used_archive


def page(
    result: dict[str, object] | None = None,
    batch: list[dict[str, object]] | None = None,
    error: str | None = None,
) -> bytes:
    result_html = ""
    if error:
        result_html = f'<div class="error">{html.escape(error)}</div>'
    elif batch:
        acceptable = sum(row["decision"] == "Acceptable culture quality" for row in batch)
        uncertain = sum(bool(row["uncertain"]) for row in batch)
        table_rows = "".join(
            f"<tr><td>{html.escape(str(row['filename']))}</td><td>{html.escape(str(row['decision']))}</td>"
            f"<td>{float(row['image_probability']) * 100:.1f}%</td><td>{float(row['context_probability']) * 100:.1f}%</td>"
            f"<td><strong>{float(row['final_probability']) * 100:.1f}%</strong></td><td>{'Yes' if row['uncertain'] else 'No'}</td></tr>"
            for row in batch
        )
        result_html = f"""<section class="results"><div class="result-head"><div><small>Batch analysis</small><h2>{len(batch)} microscopy images</h2></div>
        <div class="exports"><a href="/report.csv">Download CSV</a><a href="/report.pdf">Download PDF</a></div></div>
        <div class="scores"><article class="score-card primary"><small>Acceptable</small><strong>{acceptable}</strong><span>images</span></article>
        <article class="score-card"><small>Review recommended</small><strong>{len(batch)-acceptable}</strong><span>images</span></article>
        <article class="score-card"><small>Uncertain</small><strong>{uncertain}</strong><span>manual review priority</span></article></div>
        <div class="table-wrap"><table><thead><tr><th>Image</th><th>Decision</th><th>Image</th><th>Context</th><th>Final</th><th>Uncertain</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>"""
    elif result:
        lower_review = max(0.0, float(result["threshold"]) - float(result["review_margin"]))
        upper_review = min(1.0, float(result["threshold"]) + float(result["review_margin"]))
        uncertainty = (
            '<p class="warning">This result falls inside the evidence-based review band; expert review is recommended.</p>'
            if result["uncertain"]
            else ""
        )
        result_html = f"""
        <section class="results">
          <div class="result-head"><div><small>{html.escape(str(result['assessment_mode']))}</small><h2>{html.escape(str(result['decision']))}</h2></div>
          <span class="threshold">Manual review band {lower_review * 100:.1f}–{upper_review * 100:.1f}%</span></div>
          <div class="scores">{score_card('Microscopy image only', float(result['image_probability']))}
          {score_card('Image + experimental context', float(result['context_probability']), primary=True)}</div>{uncertainty}
          <div class="images"><figure><img src="{result['preview']}"><figcaption>Uploaded bright-field image</figcaption></figure>
          <figure><img src="{result['explanation']}"><figcaption>Red regions support acceptable quality; blue regions support review.</figcaption></figure></div>
          <div class="signal-grid"><div><h3>Image evidence</h3><ul>{signal_list(result['image_signals'])}</ul></div>
          <div><h3>Experimental context</h3><ul>{signal_list(result['context_signals'])}</ul></div></div>
        </section>"""

    cell_options = "".join(f'<option value="{item}">{item}</option>' for item in CELL_TYPES)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>CellScope AI</title>
    <style>
    :root{{--ink:#14221d;--green:#1c9b63;--deep:#0c5b3a;--paper:#f2f6f3;--line:#d8e4dd;--warn:#9a5b12}}
    *{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#eef5f0,#f8faf9);color:var(--ink);font:16px/1.5 Arial,sans-serif}}
    main{{max-width:1120px;margin:auto;padding:44px 24px 60px}}header{{margin-bottom:26px}}.eyebrow{{color:var(--green);font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
    h1{{font-size:44px;line-height:1.1;margin:5px 0 9px}}header p{{max-width:760px;color:#52645b;margin:0}}
    .card,.results{{background:#fff;border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 12px 34px #173c2b12}}
    form{{display:grid;gap:20px}}label{{display:grid;gap:7px;font-size:13px;font-weight:750;color:#42564b}}input,select{{width:100%;padding:12px 13px;border:1px solid #b9cbc1;border-radius:10px;background:#fff;color:var(--ink);font-size:15px}}
    input[type=file]{{padding:15px;border-style:dashed}}.fields{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}.upload-row{{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:end}}
    button{{border:0;background:var(--green);color:#fff;font-weight:800;padding:14px 24px;border-radius:10px;cursor:pointer;min-height:52px}}button:hover{{background:var(--deep)}}
    .hint{{margin:0;color:#687970;font-size:13px}}.results{{margin-top:22px}}.result-head{{display:flex;justify-content:space-between;gap:16px;align-items:center}}.result-head h2{{margin:3px 0;font-size:28px}}.threshold{{background:#edf4f0;padding:7px 11px;border-radius:999px;font-size:13px}}
    .scores{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}}.score-card{{border:1px solid var(--line);border-radius:14px;padding:18px;display:grid;gap:6px}}.score-card.primary{{border-color:#71bd98;background:#f3fbf7}}.score-card>strong{{font-size:30px}}.score-card>span{{color:#64766d;font-size:13px}}
    .meter{{height:10px;background:#e4ece8;border-radius:9px;overflow:hidden}}.meter i{{display:block;height:100%;background:var(--green)}}.warning{{color:var(--warn);font-weight:750;background:#fff8e9;padding:11px 14px;border-radius:10px}}
    .images{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px}}figure{{margin:0}}img{{width:100%;max-height:420px;object-fit:contain;background:#101916;border-radius:12px}}figcaption{{color:#52645b;margin-top:7px;font-size:14px}}
    .signal-grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:16px}}ul{{padding:0;list-style:none;margin:0}}li{{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:9px 0}}.positive{{color:#168654}}.negative{{color:#a54a43}}
    .validation{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}}.validation article{{padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:#f8fbf9}}.validation strong{{display:block;font-size:22px}}.validation small{{color:#607168}}
    .exports{{display:flex;gap:8px}}.exports a{{background:#eaf5ef;color:var(--deep);padding:9px 12px;border-radius:9px;text-decoration:none;font-weight:750}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{background:#f3f7f4}}
    .error{{margin-top:20px;background:#fff0ef;color:#8c2720;padding:16px;border-radius:10px}}footer{{margin-top:24px;color:#607168;font-size:13px}}
    @media(max-width:850px){{.fields{{grid-template-columns:1fr 1fr}}}}@media(max-width:620px){{.images,.scores,.signal-grid,.upload-row,.validation{{grid-template-columns:1fr}}.fields{{grid-template-columns:1fr}}h1{{font-size:35px}}.result-head{{align-items:flex-start;flex-direction:column}}}}
    </style></head><body><main><header><div class="eyebrow">Research prototype · version 3</div><h1>CellScope AI</h1><p>Explainable quality control for organ-on-a-chip microscopy. Compare what the image says with a context-aware assessment of the experiment.</p></header>
    <section class="card"><form method="post" enctype="multipart/form-data"><div class="upload-row"><label>Bright-field microscopy images or ZIP archive<input name="images" type="file" accept="image/*,.zip" multiple required></label><button type="submit">Analyze experiment</button></div>
    <div class="fields"><label>Cell type<select name="cell_type"><option value="">Not recorded</option>{cell_options}</select></label><label>Culture day<input name="day" type="number" min="0" step="1" placeholder="e.g. 3"></label><label>Seeding density (cells/ml)<input name="seeding_density_cells_ml" inputmode="decimal" placeholder="e.g. 2600000"></label><label>Hours after seeding<input name="time_after_seeding_h" type="number" min="0" step="0.1" placeholder="e.g. 24"></label><label>Flow rate<input name="flow_rate" inputmode="decimal" placeholder="e.g. 1.66"></label></div>
    <p class="hint">Unknown fields may remain empty. Without a known cell type, the safer image-only model determines the final assessment.</p></form>
    <div class="validation"><article><strong>66.6%</strong><small>balanced accuracy on unseen acquisition sessions</small></article><article><strong>73.0%</strong><small>ROC-AUC on unseen acquisition sessions</small></article><article><strong>59.9%</strong><small>balanced accuracy with a complete cell type held out</small></article></div></section>
    {result_html}<footer>Research quality-control signal only. Predictions require expert validation and are not clinical advice.</footer></main></body></html>"""
    return document.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    image_model = LogisticBaseline.load(ARTIFACTS / "model_image_only.npz")
    context_model = LogisticBaseline.load(ARTIFACTS / "model_image_plus_context.npz")
    config = json.loads((ARTIFACTS / "model_config.json").read_text(encoding="utf-8"))
    report_csv: bytes | None = None
    report_pdf: bytes | None = None

    def respond(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/report.csv" and self.report_csv is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="cellscope-report.csv"')
            self.send_header("Content-Length", str(len(self.report_csv)))
            self.end_headers()
            self.wfile.write(self.report_csv)
            return
        if self.path == "/report.pdf" and self.report_pdf is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="cellscope-report.pdf"')
            self.send_header("Content-Length", str(len(self.report_pdf)))
            self.end_headers()
            self.wfile.write(self.report_pdf)
            return
        self.respond(page())

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("The request must be smaller than 200 MB.")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
            )
            raw_items = form["images"]
            items = raw_items if isinstance(raw_items, list) else [raw_items]
            images, used_archive = uploaded_images(items)
            context = {
                name: form.getfirst(name, "")
                for name in (
                    "cell_type", "day", "seeding_density_cells_ml", "time_after_seeding_h", "flow_rate"
                )
            }
            if len(images) == 1 and not used_archive:
                filename, payload = images[0]
                result = analyze(
                    io.BytesIO(payload), context, self.image_model, self.context_model,
                    self.config, filename=filename,
                )
                self.respond(page(result=result))
                return

            batch = [
                analyze(
                    io.BytesIO(payload), context, self.image_model, self.context_model,
                    self.config, filename=filename, include_visuals=False,
                )
                for filename, payload in images
            ]
            type(self).report_csv = build_csv_report(batch)
            type(self).report_pdf = build_pdf_report(batch)
            self.respond(page(batch=batch))
        except Exception as exc:
            self.respond(page(error=str(exc)), status=400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 7860), Handler)
    print("CellScope AI v3 demo: http://127.0.0.1:7860")
    server.serve_forever()


if __name__ == "__main__":
    main()
