# raglearn

Agentic RAG over financial filings. Exact figures come from a structured store
(XBRL + extracted tables); narrative comes from hybrid vector retrieval with
reranking; an agentic router picks the path per question and a verification
harness keeps answers grounded and cited.

Every pipeline stage is a pluggable adapter selected by config, so techniques can
be swapped and compared with real evaluation numbers.

## Quick start

```bash
cp .env.example .env
make up        # build + start the stack (api, ollama, qdrant, postgres, redis, minio)
curl localhost:8000/health
curl localhost:8000/config   # capability matrix: what's active and what's built
```

Backend development:

```bash
make install   # uv sync
make check     # lint + type-check + test
```

## Repository

```
backend/   FastAPI service and the RAG pipeline (see backend/README.md)
config/    layered configuration (config.yaml + config.<env>.yaml)
infra/     Dockerfile + compose
```

Requires [uv](https://docs.astral.sh/uv/) and Docker.
