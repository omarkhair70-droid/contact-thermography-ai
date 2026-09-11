FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Coolify's Docker-image health probe expects curl or wget to exist inside
# the application container. python:3.12-slim ships with neither by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-vision.txt ./
RUN pip install --no-cache-dir -r requirements-vision.txt

COPY . .
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/runtime /app/app/static /var/lib/lct/storage \
    && chown -R app:app /app/runtime /app/app/static /var/lib/lct/storage

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
