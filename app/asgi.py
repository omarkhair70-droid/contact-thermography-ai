from __future__ import annotations

import logging
import os
import threading
import time
import uuid

from app.main import app as fastapi_app


UVICORN_LOGGER = logging.getLogger("uvicorn.error")
HUMAN_LOGGER = logging.getLogger("mumguard.human_runtime")
HUMAN_LOGGER.setLevel(logging.INFO)
if UVICORN_LOGGER.handlers:
    HUMAN_LOGGER.handlers = list(UVICORN_LOGGER.handlers)
HUMAN_LOGGER.propagate = False


def _warm_dinov2_runtime() -> None:
    started = time.perf_counter()
    try:
        from app.services import dinov2_service

        dinov2_service.runtime.load_official()
    except Exception as exc:
        UVICORN_LOGGER.warning("DINOV2_WARMUP_FAILED elapsed_s=%.3f reason=%s", time.perf_counter() - started, exc)
    else:
        UVICORN_LOGGER.info("DINOV2_WARMUP_READY elapsed_s=%.3f", time.perf_counter() - started)


if os.getenv("MUMGUARD_DINOV2_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
    # Warm in the background so service readiness is not blocked by model loading.
    # Production keeps TORCH_HOME on durable storage, so the pinned weights survive
    # container restarts and the first examination does not pay the cold-load cost.
    threading.Thread(target=_warm_dinov2_runtime, name="dinov2-warmup", daemon=True).start()


class HumanRequestObserver:
    """Log the human-exam request before FastAPI parses multipart form data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") != "/api/human-exams/analyze":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_id = uuid.uuid4().hex[:10]
        started = time.perf_counter()
        status_code = None

        UVICORN_LOGGER.info(
            "HUMAN_HTTP_RECEIVED request_id=%s method=%s content_length=%s content_type=%s",
            request_id,
            scope.get("method"),
            headers.get("content-length", "unknown"),
            headers.get("content-type", "unknown"),
        )

        async def observed_send(message):
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except Exception:
            UVICORN_LOGGER.exception(
                "HUMAN_HTTP_EXCEPTION request_id=%s elapsed_s=%.3f",
                request_id,
                time.perf_counter() - started,
            )
            raise
        else:
            UVICORN_LOGGER.info(
                "HUMAN_HTTP_COMPLETED request_id=%s status=%s elapsed_s=%.3f",
                request_id,
                status_code if status_code is not None else "unknown",
                time.perf_counter() - started,
            )


app = HumanRequestObserver(fastapi_app)
