from __future__ import annotations

import os
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str, fallback: str = "item") -> str:
    value = Path(value or fallback).name
    value = _SAFE_COMPONENT.sub("_", value).strip("._")
    return value[:120] or fallback


class FilesystemStorage:
    """Durable filesystem adapter.

    In staging this root is backed by a persistent Docker volume (or an Oracle
    block-volume bind mount). The rest of the application depends on this
    adapter instead of a repository-relative generated-image directory.
    """

    backend = "filesystem"

    def __init__(self, root: Path):
        self.root = root
        self.uploads_root = root / "uploads"
        self.generated_root = root / "generated"
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.generated_root.mkdir(parents=True, exist_ok=True)

    def exam_upload_dir(self, exam_id: str) -> Path:
        path = self.uploads_root / _safe_component(exam_id, "exam")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def generated_exam_dir(self, exam_id: str) -> Path:
        path = self.generated_root / _safe_component(exam_id, "exam")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist_upload(self, exam_id: str, source_name: str, payload: bytes) -> Path:
        filename = _safe_component(source_name, "upload.bin")
        destination = self.exam_upload_dir(exam_id) / f"{uuid.uuid4().hex[:10]}_{filename}"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        return destination

    def generated_url(self, exam_id: str, *parts: str) -> str:
        safe_parts = [_safe_component(exam_id, "exam")]
        safe_parts.extend(_safe_component(part) for part in parts)
        return "/static/generated/" + "/".join(safe_parts)

    def healthcheck(self) -> bool:
        return (
            self.root.exists()
            and self.uploads_root.exists()
            and self.generated_root.exists()
            and os.access(self.root, os.W_OK | os.X_OK)
        )


def _build_storage():
    backend = os.getenv("STORAGE_BACKEND", "filesystem").strip().lower()
    if backend != "filesystem":
        raise RuntimeError(
            f"Unsupported STORAGE_BACKEND={backend!r}. "
            "Use filesystem backed by persistent Oracle storage for this release."
        )
    root = Path(os.getenv("STORAGE_ROOT", str(ROOT / "runtime" / "storage"))).expanduser()
    return FilesystemStorage(root.resolve())


storage = _build_storage()
