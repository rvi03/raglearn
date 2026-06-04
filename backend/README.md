# raglearn backend

Agentic RAG over financial filings: exact figures from a structured store,
narrative from hybrid vector retrieval, behind one agentic router.

## Layout

```
src/raglearn/
  core/          config, registry, interfaces, domain types, logging, errors
  ingestion/     connectors, intake, parsers, chunkers, embedders
  retrieval/     transforms, router, retrievers, reranker, fusion
  generation/    llm backends
  harness/       answer-quality verification steps (grows over time)
  stores/        qdrant, duckdb, graph clients
  cost/          cost models (local = $0)
  observability/ security/ eval/
  api/           FastAPI app, routes, SSE
tests/
```

## How adapters work

Each pipeline stage is a `Protocol` in `core/interfaces`. Implementations
register against the process registry with a decorator next to the class:

```python
@registry.register("embedder", "bge_m3")
class BgeM3Embedder: ...
```

A stage subpackage's `__init__` imports its adapter modules so those decorators
run; `core/bootstrap.py` imports every subpackage at startup. Config selects the
active adapter per stage; `/config` serves the resulting capability matrix.

## Develop

```
make install     # uv sync
make check       # lint + type-check + test
make test        # pytest
```

Requires [uv](https://docs.astral.sh/uv/). Config is read from `../config`
(override with `RAGLEARN_CONFIG_DIR`); the profile is `RAGLEARN_ENV` (`dev`).
