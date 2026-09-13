FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app/runtime \
    TORCH_HOME=/var/lib/lct/storage/torch-cache \
    XDG_CACHE_HOME=/var/lib/lct/storage/.cache \
    MUMGUARD_DINOV2_WARMUP=1 \
    DINOV2_BATCH_SIZE=4

WORKDIR /app

# Coolify's Docker-image health probe expects curl or wget to exist inside
# the application container. python:3.12-slim ships with neither by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-vision.txt ./

# Production runs on the CPU-only Oracle/Coolify host. Installing torch from
# the default PyPI index pulls multi-gigabyte CUDA/NVIDIA dependencies on
# Linux ARM64, which are unused here and can exhaust the host disk during
# image export. Keep the normal app dependencies from PyPI, then install the
# official CPU-only PyTorch wheels explicitly.
RUN pip install --no-cache-dir -r requirements.txt 'pillow>=10' \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       'torch==2.14.0+cpu' 'torchvision==0.29.0+cpu'

COPY . .
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p \
       /app/runtime \
       /app/app/static \
       /var/lib/lct/storage \
       /var/lib/lct/storage/torch-cache \
       /var/lib/lct/storage/.cache \
    && chown -R app:app /app/runtime /app/app/static /var/lib/lct/storage

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
