# Oracle staging rollback

Rollback is intentionally Git-SHA driven and preserves the PostgreSQL and media volumes.

## Before each deployment

Record the known-good SHA:

```bash
git rev-parse HEAD
```

Create a database backup:

```bash
mkdir -p backups
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml \
  exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' \
  | gzip > "backups/postgres-$(date -u +%Y%m%dT%H%M%SZ).sql.gz"
```

Create a media-volume backup:

```bash
docker run --rm \
  -v contact-thermography-ai-staging-media:/data:ro \
  -v "$PWD/backups:/backup" \
  alpine:3.20 sh -c 'tar -czf /backup/media-$(date -u +%Y%m%dT%H%M%SZ).tar.gz -C /data .'
```

## Application rollback

Do not delete volumes. Check out the last known-good SHA and recreate only the software containers:

```bash
git fetch origin
git checkout <KNOWN_GOOD_SHA>

docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml build
docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml up -d --remove-orphans
python deploy/smoke.py "https://$(grep '^APP_DOMAIN=' deploy/staging.env | cut -d= -f2-)"
```

Never use `docker compose down -v` as part of an ordinary rollback; `-v` deletes the persistent database/media volumes.

## Data restore only when required

This lane does not introduce a destructive schema migration, so a normal code rollback should keep the current data. Restore backups only if data itself was damaged or a future migration explicitly requires it.

PostgreSQL restore example:

```bash
gzip -dc backups/<postgres-backup>.sql.gz | \
  docker compose --env-file deploy/staging.env -f deploy/docker-compose.staging.yml \
  exec -T db sh -c 'psql -U "$POSTGRES_USER" "$POSTGRES_DB"'
```

Media restore example (after deliberately confirming the target contents may be replaced):

```bash
docker run --rm \
  -v contact-thermography-ai-staging-media:/data \
  -v "$PWD/backups:/backup:ro" \
  alpine:3.20 sh -c 'rm -rf /data/* /data/.[!.]* /data/..?* 2>/dev/null || true; tar -xzf /backup/<media-backup>.tar.gz -C /data'
```

After any rollback or restore, run `deploy/smoke.py` and inspect `docker compose ... ps` before declaring staging healthy.
