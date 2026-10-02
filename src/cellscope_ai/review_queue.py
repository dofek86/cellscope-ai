"""Local, opt-in review queue for new acquisition sessions."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data/review_queue"


def split_for_session(session_id: str) -> str:
    """Freeze a session's training/evaluation assignment before labels arrive."""
    if not session_id or not session_id.strip() or len(session_id) > 100:
        raise ValueError("Vnesi oznako zajemne seje (največ 100 znakov).")
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 5
    return "locked_evaluation" if bucket == 0 else "candidate_training"


def review_priority(probability: float, base: float, patch: float) -> float:
    """Selection heuristic only; it is not a biological reason for rejection."""
    return (1 - 2 * abs(probability - .5)) + abs(base - patch)


class ReviewQueue:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "images").mkdir(exist_ok=True)
        self.database = self.directory / "reviews.sqlite3"
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                split TEXT NOT NULL,
                original_name TEXT NOT NULL,
                probability_good REAL NOT NULL,
                base_probability_good REAL NOT NULL,
                patch_probability_good REAL NOT NULL,
                priority REAL NOT NULL,
                texture_review_flag INTEGER,
                texture_review_stability REAL,
                created_at TEXT NOT NULL,
                label TEXT,
                reviewer TEXT,
                expert_confirmed INTEGER NOT NULL DEFAULT 0,
                note TEXT,
                labeled_at TEXT
            )""")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(images)")}
            if "texture_review_flag" not in columns:
                connection.execute("ALTER TABLE images ADD COLUMN texture_review_flag INTEGER")
            if "texture_review_stability" not in columns:
                connection.execute("ALTER TABLE images ADD COLUMN texture_review_stability REAL")

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, data: bytes, name: str, session_id: str, result: dict) -> dict:
        split = split_for_session(session_id)
        image_id = hashlib.sha256(data).hexdigest()
        with self.connect() as connection:
            prior = connection.execute("SELECT session_id FROM images WHERE image_id = ?",
                                       (image_id,)).fetchone()
            if prior is not None:
                if prior["session_id"] != session_id:
                    raise ValueError("Enaka slika je že shranjena v drugi seji; "
                                     "da preprečimo uhajanje podatkov, je ne dodamo.")
                return {"image_id": image_id, "split": split, "already_saved": True}
            image_path = self.directory / "images" / f"{image_id}.img"
            thumbnail_path = self.directory / "images" / f"{image_id}.jpg"
            image_path.write_bytes(data)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((360, 260), Image.Resampling.LANCZOS)
                    image.save(thumbnail_path, format="JPEG", quality=80)
                base = float(result["base_probability_good"])
                patch = float(result["patch_probability_good"])
                probability = float(result["probability_good"])
                priority = review_priority(probability, base, patch)
                texture_flag = result.get("texture_review_flag")
                if texture_flag is not None and not isinstance(texture_flag, bool):
                    raise TypeError("Texture review flag must be boolean or unavailable")
                stability = result.get("texture_review_stability")
                if stability is not None:
                    stability = float(stability)
                    if not 0 <= stability <= 1:
                        raise ValueError("Texture context stability must be in [0, 1]")
                connection.execute("""INSERT INTO images (
                    image_id, session_id, split, original_name, probability_good,
                    base_probability_good, patch_probability_good, priority,
                    texture_review_flag, texture_review_stability, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (image_id, session_id, split, Path(name).name, probability,
                     base, patch, priority,
                     int(texture_flag) if texture_flag is not None else None,
                     stability,
                     datetime.now(UTC).isoformat()))
            except Exception:
                image_path.unlink(missing_ok=True)
                thumbnail_path.unlink(missing_ok=True)
                raise
        return {"image_id": image_id, "split": split, "already_saved": False}

    def list_items(self, limit: int = 200) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("""SELECT image_id, session_id, split,
                original_name, probability_good, base_probability_good,
                patch_probability_good, priority, texture_review_flag,
                texture_review_stability,
                created_at, label, reviewer,
                expert_confirmed, note, labeled_at FROM images
                ORDER BY (label IS NULL) DESC, priority DESC, created_at DESC
                LIMIT ?""", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def set_label(self, image_id: str, label: str, reviewer: str,
                  expert_confirmed: bool, note: str = "") -> None:
        if label not in {"good", "bad", "unsure"}:
            raise ValueError("Oznaka mora biti good, bad ali unsure.")
        if not reviewer.strip() or len(reviewer) > 100 or len(note) > 1000:
            raise ValueError("Vnesi ime pregledovalca; opomba ima največ 1000 znakov.")
        if not isinstance(expert_confirmed, bool):
            raise TypeError("Strokovna potrditev mora biti da ali ne.")
        if label == "unsure" and expert_confirmed:
            raise ValueError("Negotove oznake ni mogoče potrditi za učenje.")
        with self.connect() as connection:
            changed = connection.execute("""UPDATE images SET label = ?, reviewer = ?,
                expert_confirmed = ?, note = ?, labeled_at = ? WHERE image_id = ?""",
                (label, reviewer.strip(), int(expert_confirmed), note.strip(),
                 datetime.now(UTC).isoformat(), image_id)).rowcount
        if changed != 1:
            raise ValueError("Slika ni v čakalni vrsti.")

    def training_status(self) -> dict:
        with self.connect() as connection:
            rows = connection.execute("""SELECT split, label, COUNT(*) AS images,
                COUNT(DISTINCT session_id) AS sessions FROM images
                WHERE expert_confirmed = 1 AND label IN ('good', 'bad')
                GROUP BY split, label""").fetchall()
        counts = {split: {label: {"images": 0, "sessions": 0}
                          for label in ("good", "bad")}
                  for split in ("candidate_training", "locked_evaluation")}
        for row in rows:
            counts[row["split"]][row["label"]] = {
                "images": row["images"], "sessions": row["sessions"]}
        return counts
