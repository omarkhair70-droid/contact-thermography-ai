from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main
from app.services import analysis_engine, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "lct-test.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.connect().close()

    generated = tmp_path / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(analysis_engine, "STATIC_GENERATED", generated)

    runtime_root = tmp_path / "runtime-root"
    (runtime_root / "app" / "static" / "generated").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "ROOT", runtime_root)

    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def sample_png():
    img = np.zeros((180, 180, 3), dtype=np.uint8)
    for y in range(img.shape[0]):
        img[y, :, 0] = (20 + y) % 255
        img[y, :, 1] = (80 + 2 * y) % 255
        img[y, :, 2] = (140 + y // 2) % 255
    cv2.circle(img, (90, 90), 55, (20, 220, 170), 4)
    cv2.line(img, (45, 90), (135, 90), (220, 60, 30), 5)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()
