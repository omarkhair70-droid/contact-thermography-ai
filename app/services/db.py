from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "runtime"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "lct.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    exam_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    clinical_claim TEXT NOT NULL,
    source_images INTEGER NOT NULL,
    plates_detected INTEGER NOT NULL,
    bilateral_pairs_created INTEGER NOT NULL,
    result_json TEXT NOT NULL
)
"""


def _database_url():
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        if explicit.startswith("postgres://"):
            return "postgresql+psycopg://" + explicit[len("postgres://"):]
        if explicit.startswith("postgresql://"):
            return "postgresql+psycopg://" + explicit[len("postgresql://"):]
        return explicit

    host = os.getenv("DB_HOST", "").strip()
    if host:
        password = os.getenv("DB_PASSWORD")
        if password is None:
            raise RuntimeError("DB_PASSWORD is required when DB_HOST is configured")
        return URL.create(
            "postgresql+psycopg",
            username=os.getenv("DB_USER", "lct"),
            password=password,
            host=host,
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "lct"),
        )

    return f"sqlite:///{DB_PATH}"


DATABASE_URL = _database_url()
_ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args={"check_same_thread": False} if str(DATABASE_URL).startswith("sqlite") else {},
)


def init_db():
    with _ENGINE.begin() as conn:
        conn.execute(text(SCHEMA))


def database_backend():
    return _ENGINE.url.get_backend_name()


def database_health():
    try:
        with _ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def save_exam(result: dict):
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    params = {
        "exam_id": result["exam_id"],
        "created_at": now,
        "analysis_type": result["analysis_type"],
        "clinical_claim": result["clinical_claim"],
        "source_images": int(result["source_images"]),
        "plates_detected": int(result["plates_detected"]),
        "bilateral_pairs_created": int(result["bilateral_pairs_created"]),
        "result_json": payload,
    }
    with _ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO exams (
                  exam_id, created_at, analysis_type, clinical_claim,
                  source_images, plates_detected, bilateral_pairs_created, result_json
                )
                VALUES (
                  :exam_id, :created_at, :analysis_type, :clinical_claim,
                  :source_images, :plates_detected, :bilateral_pairs_created, :result_json
                )
                ON CONFLICT(exam_id) DO UPDATE SET
                  created_at=excluded.created_at,
                  analysis_type=excluded.analysis_type,
                  clinical_claim=excluded.clinical_claim,
                  source_images=excluded.source_images,
                  plates_detected=excluded.plates_detected,
                  bilateral_pairs_created=excluded.bilateral_pairs_created,
                  result_json=excluded.result_json
                """
            ),
            params,
        )
    return now


def get_exam(exam_id: str):
    with _ENGINE.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM exams WHERE exam_id=:exam_id"),
            {"exam_id": exam_id},
        ).mappings().first()
    if row is None:
        return None
    item = dict(row)
    item["result"] = json.loads(item.pop("result_json"))
    return item


def list_exams(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    with _ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT exam_id, created_at, analysis_type, clinical_claim,
                       source_images, plates_detected, bilateral_pairs_created
                FROM exams ORDER BY created_at DESC LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
    return [dict(row) for row in rows]


init_db()
