from __future__ import annotations

import logging
import os
import threading
import time
import uuid

import numpy as np

from app.main import app as fastapi_app


UVICORN_LOGGER = logging.getLogger("uvicorn.error")
HUMAN_LOGGER = logging.getLogger("mumguard.human_runtime")
HUMAN_LOGGER.setLevel(logging.INFO)
if UVICORN_LOGGER.handlers:
    HUMAN_LOGGER.handlers = list(UVICORN_LOGGER.handlers)
HUMAN_LOGGER.propagate = False


def _warm_dinov2_runtime(*, fail_hard: bool = False) -> None:
    """Load DINO and execute the same patch path used by live human exams.

    Loading model weights alone was not sufficient on the CPU-only Oracle host:
    the first real `forward_features` call could still pay a large one-time cost.
    Production therefore warms a representative batch before the ASGI app is
    exposed. Blocking warmup fails closed so a broken warmup never becomes a
    client-visible 2-3 minute first examination.
    """
    started = time.perf_counter()
    warm_batch = 1
    try:
        from app.services import dinov2_service
        from app.services.dinov2_patch_encoder import encode_patches_batch

        dinov2_service.runtime.load_official()
        try:
            configured = int(os.getenv("DINOV2_BATCH_SIZE", "4"))
        except (TypeError, ValueError):
            configured = 4
        warm_batch = max(1, min(configured, 4))
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        encode_patches_batch(
            [frame.copy() for _ in range(warm_batch)],
            batch_size=warm_batch,
        )
    except Exception as exc:
        UVICORN_LOGGER.warning(
            "DINOV2_WARMUP_FAILED elapsed_s=%.3f reason=%s",
            time.perf_counter() - started,
            exc,
        )
        if fail_hard:
            raise
    else:
        UVICORN_LOGGER.info(
            "DINOV2_WARMUP_READY elapsed_s=%.3f inference_batch=%d",
            time.perf_counter() - started,
            warm_batch,
        )


if os.getenv("MUMGUARD_DINOV2_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
    warmup_mode = os.getenv("MUMGUARD_DINOV2_WARMUP_MODE", "background").strip().lower()
    if warmup_mode == "blocking":
        # Production pays both model-load and first-forward costs once during
        # container startup, before a client can submit an examination.
        _warm_dinov2_runtime(fail_hard=True)
    else:
        # Background mode remains available for development environments.
        threading.Thread(
            target=_warm_dinov2_runtime,
            name="dinov2-warmup",
            daemon=True,
        ).start()


class HumanRequestObserver:
    """Observe upload-body and total request time without buffering the request.

    This keeps the earlier timeout investigation measurable: a long upload/body
    phase and a long model-compute phase are separate problems and should not be
    guessed from one final request duration.
    """

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
        body_bytes = 0
        body_complete_logged = False

        UVICORN_LOGGER.info(
            "HUMAN_HTTP_RECEIVED request_id=%s method=%s content_length=%s content_type=%s",
            request_id,
            scope.get("method"),
            headers.get("content-length", "unknown"),
            headers.get("content-type", "unknown"),
        )

        async def observed_receive():
            nonlocal body_bytes, body_complete_logged
            message = await receive()
            if message.get("type") == "http.request":
                body_bytes += len(message.get("body", b""))
                if not message.get("more_body", False) and not body_complete_logged:
                    body_complete_logged = True
                    UVICORN_LOGGER.info(
                        "HUMAN_BODY_COMPLETE request_id=%s bytes=%d elapsed_s=%.3f",
                        request_id,
                        body_bytes,
                        time.perf_counter() - started,
                    )
            return message

        async def observed_send(message):
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, observed_receive, observed_send)
        except Exception:
            UVICORN_LOGGER.exception(
                "HUMAN_HTTP_EXCEPTION request_id=%s bytes=%d elapsed_s=%.3f",
                request_id,
                body_bytes,
                time.perf_counter() - started,
            )
            raise
        else:
            UVICORN_LOGGER.info(
                "HUMAN_HTTP_COMPLETED request_id=%s status=%s bytes=%d elapsed_s=%.3f",
                request_id,
                status_code if status_code is not None else "unknown",
                body_bytes,
                time.perf_counter() - started,
            )


app = HumanRequestObserver(fastapi_app)
