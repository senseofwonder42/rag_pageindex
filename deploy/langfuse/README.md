# Langfuse self-hosted stack

Docker Compose stack that runs [Langfuse](https://langfuse.com) v3 locally so
LLM/VLM traces from `rag_pageindex` stay on this machine. Adapted verbatim
from the upstream [reference compose file](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml).

The stack is six containers: `langfuse-web`, `langfuse-worker`, `postgres`,
`clickhouse`, `redis`, `minio`. Expect ~2 GB resident memory at idle.

## First-time setup

1. Create your secrets file:
   ```sh
   cp deploy/langfuse/.env.example deploy/langfuse/.env
   $EDITOR deploy/langfuse/.env   # generate values per the comments
   ```

2. Bring the stack up:
   ```sh
   docker compose -f deploy/langfuse/docker-compose.yml --env-file deploy/langfuse/.env up -d
   ```

3. Open <http://localhost:3000> and sign up (the first user becomes the org
   admin). Create a project, then `Settings → API Keys → Create new API keys`.

4. Wire the keys into the app's `.env` (NOT the Langfuse `.env`):
   ```sh
   tracing_enabled=true
   langfuse_host=http://localhost:3000
   langfuse_public_key=pk-lf-...
   langfuse_secret_key=sk-lf-...
   ```

5. Run `uv run start-app --pdf-path <file>`. Refresh the Langfuse UI; you
   should see one trace per run with nested spans for each pipeline stage and
   one `generation` per LLM/VLM call.

## Optional: bootstrap a project on first boot

Uncomment the `LANGFUSE_INIT_*` block in `.env` to skip the UI-based signup
and have the web container create the org, project, user, and API keys on
startup. Useful for ephemeral dev environments.

## Stopping / wiping

```sh
docker compose -f deploy/langfuse/docker-compose.yml down           # stop, keep data
docker compose -f deploy/langfuse/docker-compose.yml down -v        # stop, wipe all data
```

## Ports

| Port | Service                    | Notes |
|------|----------------------------|-------|
| 3000 | langfuse-web (UI + API)    | exposed on all interfaces |
| 9090 | minio (S3 API for media)   | exposed on all interfaces |
| 9091 | minio console (loopback)   | <http://127.0.0.1:9091> |
| 5432 | postgres (loopback)        | |
| 8123 | clickhouse HTTP (loopback) | |
| 9000 | clickhouse native (loopback) | |
| 6379 | redis (loopback)           | |
| 3030 | langfuse-worker (loopback) | health/internal |
