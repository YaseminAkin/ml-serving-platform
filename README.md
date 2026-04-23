# ML Serving Platform

Production-grade ML model serving platform with a FastAPI gateway, model registry, and evaluation harness.

## Components

| Component | Description |
|-----------|-------------|
| `gateway/` | FastAPI inference gateway with auth, caching, and routing |
| `registry/` | Model version and artifact registry |
| `eval/` | Offline and online evaluation harness |
| `db/` | Database migrations (Alembic) |
| `infra/` | Docker Compose, Dockerfile, Grafana dashboards |

## Quick Start

```bash
# Install dependencies
make install-dev

# Start infrastructure
make up

# Run the gateway (dev)
uvicorn gateway.main:app --reload
```

## Development

```bash
make lint        # ruff check
make format      # black + ruff --fix
make typecheck   # mypy
make test        # pytest
make test-cov    # pytest with coverage
```

## Requirements

- Python 3.11+
- Docker + Docker Compose
- Redis 7+
- PostgreSQL 16+
