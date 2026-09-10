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
