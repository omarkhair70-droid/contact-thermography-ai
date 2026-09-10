from __future__ import annotations
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone

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
);
"""

def connect():
    conn=sqlite3.connect(DB_PATH)
    conn.row_factory=sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)
    return conn

def save_exam(result: dict):
    now=datetime.now(timezone.utc).isoformat()
    payload=json.dumps(result, separators=(",",":"), ensure_ascii=False, allow_nan=False)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO exams (
              exam_id, created_at, analysis_type, clinical_claim,
              source_images, plates_detected, bilateral_pairs_created, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exam_id) DO UPDATE SET
              created_at=excluded.created_at,
              analysis_type=excluded.analysis_type,
              clinical_claim=excluded.clinical_claim,
              source_images=excluded.source_images,
              plates_detected=excluded.plates_detected,
              bilateral_pairs_created=excluded.bilateral_pairs_created,
              result_json=excluded.result_json
            """,
            (
                result["exam_id"], now, result["analysis_type"], result["clinical_claim"],
                int(result["source_images"]), int(result["plates_detected"]),
                int(result["bilateral_pairs_created"]), payload
            ),
        )
    return now

def get_exam(exam_id: str):
    with connect() as conn:
        row=conn.execute("SELECT * FROM exams WHERE exam_id=?", (exam_id,)).fetchone()
    if row is None:
        return None
    item=dict(row)
    item["result"]=json.loads(item.pop("result_json"))
    return item

def list_exams(limit: int=50):
    limit=max(1,min(int(limit),200))
    with connect() as conn:
        rows=conn.execute(
            """
            SELECT exam_id, created_at, analysis_type, clinical_claim,
                   source_images, plates_detected, bilateral_pairs_created
            FROM exams ORDER BY created_at DESC LIMIT ?
            """,
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

connect().close()
