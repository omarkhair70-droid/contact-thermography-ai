# Lane: Oracle Deployment

Branch: `feat/oracle-deploy`

Goal: produce a repeatable staging deployment suitable for a client demo.

Acceptance criteria:
- Docker deployment on Oracle.
- PostgreSQL persistence in staging/production config while local SQLite remains possible for development.
- Durable storage path/object-storage abstraction for uploaded/generated images.
- Healthcheck and restart policy.
- Environment-variable configuration with no secrets committed.
- Reverse proxy / HTTPS-ready configuration.
- Deployment and rollback docs.
- Do not modify model semantics or UI beyond deployment necessities.

## Lane C implementation

- `app/services/db.py` now uses SQLAlchemy so the existing exam persistence contract works with SQLite locally and PostgreSQL in staging.
- `app/services/storage.py` owns durable upload/generated-media paths. The current backend is filesystem, designed to sit on a persistent Oracle/Docker volume; generated URLs keep the existing `/static/generated/...` shape.
- Valid original uploads are retained under the durable storage root; uploaded originals are not publicly served.
- `deploy/docker-compose.staging.yml` provides PostgreSQL, the FastAPI app, and Caddy with dependency health checks and `restart: unless-stopped`.
- `deploy/Caddyfile` is ready for automatic HTTPS when `APP_DOMAIN` resolves to the Oracle VM.
- Secrets are supplied only through the host-local `deploy/staging.env`, which is ignored by Git.
- Deployment, backup, and rollback procedures are in `docs/ORACLE_STAGING_DEPLOYMENT.md` and `docs/ORACLE_ROLLBACK.md`.
- `deploy/smoke.py` checks health, persistence API reachability, storage/database backend reporting, and dashboard reachability.

## Validation performed for this lane

On 2026-09-10 the implementation was checked with Python bytecode compilation, a fresh SQLite save/get/list persistence smoke, a filesystem upload/generated-media storage smoke, and static YAML/structure checks for both Compose files. Those checks passed.

The execution environment used for this lane did not provide a Docker daemon, so an actual `docker compose up` and public Oracle HTTPS smoke could not be executed here. The exact Oracle-host commands and the post-deploy smoke command are documented for integration/deployment review.
