# Oracle staging deployment

This lane deploys the existing FastAPI application without changing AI semantics or redesigning the UI. Staging uses PostgreSQL, durable media storage, Caddy as the HTTPS reverse proxy, health checks, and restart policies.

## Oracle host prerequisites

Use an Oracle Linux or Ubuntu compute VM with a public IP, DNS A/AAAA record for the staging hostname, and inbound TCP 80/443 plus UDP 443. Install Docker Engine with the Docker Compose v2 plugin. Keep port 8000 closed publicly; only Caddy reaches the application over the Compose network.

For stronger durability, place Docker's data root on an attached Oracle Block Volume. The named volumes used here survive container recreation and `docker compose down`; do not use `down -v` during normal deploys or rollbacks.

## First deployment

```bash
git clone https://github.com/omarkhair70-droid/contact-thermography-ai.git
cd contact-thermography-ai
git fetch --all --prune
git checkout integration
cp deploy/staging.env.example deploy/staging.env
chmod 600 deploy/staging.env
# Edit APP_DOMAIN and POSTGRES_PASSWORD in deploy/staging.env.

docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml config >/dev/null
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml build --pull
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml up -d

docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml ps
python deploy/smoke.py "https://$(grep '^APP_DOMAIN=' deploy/staging.env | cut -d= -f2-)"
```

Caddy obtains and renews HTTPS certificates automatically after the hostname resolves to the VM and ports 80/443 are reachable.

## Persistence model

Exam records use PostgreSQL in staging. The application constructs its PostgreSQL connection from `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`; this avoids committing or assembling a password-bearing URL in source control. Local development still supports `DATABASE_URL=sqlite:///...`.

Valid uploaded source images are persisted below the storage adapter's `uploads/` tree. Generated plates, masks, and bilateral images are persisted below `generated/`. Staging maps the storage root to the named volume `contact-thermography-ai-staging-media` and serves only generated images under the existing `/static/generated/...` URL shape. Uploaded originals are not publicly mounted.

## Secrets and configuration

`deploy/staging.env` is host-local and ignored by Git. Never commit it. Rotate the PostgreSQL password by changing the host file and the database role password together. No cloud, database, TLS, or application secrets are required in the image.

## Update deployment

Before every update, record the currently deployed Git SHA and take backups described in `docs/ORACLE_ROLLBACK.md`. Then:

```bash
git fetch origin
git checkout integration
git pull --ff-only origin integration

docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml build
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml up -d --remove-orphans
python deploy/smoke.py "https://$(grep '^APP_DOMAIN=' deploy/staging.env | cut -d= -f2-)"
```

## Operational checks

```bash
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml ps
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml logs --tail=200 lct
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml logs --tail=200 db
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml logs --tail=200 proxy
```

`GET /health` returns HTTP 200 only when the process is up and reports the configured database/storage backends. Docker also probes this endpoint from inside the application container. PostgreSQL has its own `pg_isready` health check, and all three services use `restart: unless-stopped`.
