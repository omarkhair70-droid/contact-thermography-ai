# Oracle shared-host staging mode

Use this mode when the target Oracle host already runs an edge proxy or other services on ports 80/443. It keeps Contact Thermography AI isolated behind a configurable high port instead of starting a second Caddy instance.

The standalone deployment in `deploy/docker-compose.staging.yml` remains appropriate for a dedicated VM where this project owns ports 80/443.

## What this mode runs

- PostgreSQL 16 in a private Compose network.
- The FastAPI application with persistent media storage.
- No bundled reverse proxy.
- Application published on `LCT_BIND_ADDRESS:LCT_PORT` (default `127.0.0.1:8110`).

## Prepare the host

```bash
git clone https://github.com/omarkhair70-droid/contact-thermography-ai.git
cd contact-thermography-ai
git fetch --all --prune
git checkout integration

cp deploy/shared-host.env.example deploy/shared-host.env
chmod 600 deploy/shared-host.env
# Set a long random POSTGRES_PASSWORD before starting.
```

Do not deploy the pre-integration lane branch as the client candidate. The final deployment should be made from the green `integration` SHA after all feature PRs are reconciled.

## Start on the isolated port

```bash
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml config >/dev/null
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml build --pull
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml up -d

docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml ps
python deploy/smoke.py http://127.0.0.1:8110
```

If the host reverse proxy is on the same machine, keep `LCT_BIND_ADDRESS=127.0.0.1` and route the chosen staging hostname to `http://127.0.0.1:8110`.

If the reverse proxy is on another trusted machine, change `LCT_BIND_ADDRESS` to a private/tailnet-reachable address (or `0.0.0.0` only when necessary), and restrict the host firewall/security rules so the high port is reachable only from the trusted proxy path. Do not expose port 8110 broadly to the public Internet.

## Existing reverse-proxy target

The upstream target is simply:

```text
http://<LCT_HOST>:8110
```

The public proxy should terminate HTTPS and forward normal HTTP/WebSocket traffic to that upstream. The application itself does not require a public port 8000/8110 when the proxy is correctly configured.

## Update after final integration

```bash
git fetch origin
git checkout integration
git pull --ff-only origin integration

docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml build
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml up -d --remove-orphans
python deploy/smoke.py http://127.0.0.1:8110
```

Then run the same smoke check through the public HTTPS hostname.

## Rollback

Record the known-good integration SHA before each update. To roll back software without deleting data:

```bash
git checkout <KNOWN_GOOD_SHA>
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml build
docker compose --env-file deploy/shared-host.env -f deploy/docker-compose.shared-host.yml up -d --remove-orphans
python deploy/smoke.py http://127.0.0.1:8110
```

Never use `docker compose down -v` for a normal rollback because that removes the named PostgreSQL/media volumes.

## Integration gate before the client URL

Do not call the staging URL demo-ready until:

1. Lane A live DINOv2 is merged into `integration`.
2. Product UI, Oracle and QA lanes are reconciled.
3. The full regression suite is green on the integration candidate.
4. Local/high-port smoke passes.
5. Public HTTPS smoke passes through the chosen reverse proxy.
6. `clinical_claim` remains `NONE` and TLC/device provenance is preserved in upload, history and report paths.
