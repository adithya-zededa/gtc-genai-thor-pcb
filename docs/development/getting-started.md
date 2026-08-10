# Development Guide

## Prerequisites

- Python 3.11+
- (Optional) Docker & Docker Compose for containerised runs
- (Optional) NVIDIA GPU + CUDA for local vLLM inference

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd gtc-genai-thor-pcb

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dev dependencies (includes testing & linting)
pip install -r requirements/dev.txt

# Point at a vLLM server (required — the app has no built-in inference)
export VLLM_URL=http://localhost:8000

# Optional: edit a copy of the config instead of the tracked file.
# CAMERA_AGENT_CONFIG is what selects it — copying alone does nothing.
cp config.yaml config.local.yaml
export CAMERA_AGENT_CONFIG=config.local.yaml

# Run the development server
python run.py
```

Settings precedence is **environment > `config.yaml` > default**. The
environment is parsed in one place, [`core/config.py`](../../core/config.py).

## Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/

# With coverage
pytest --cov=agents --cov=services --cov=app tests/
```

## Code Style

The project uses **Black** for formatting and **isort** for import sorting:

```bash
black .
isort .
flake8 .
mypy agents/ services/ app/ core/
```

## Project Layout Conventions

| Directory        | What belongs here                          |
|----------------- |------------------------------------------- |
| `agents/core/`   | Monitoring loop, detection agent, state     |
| `agents/tools/`  | Callable tools (each file = one domain)    |
| `agents/conversation/` | The chat turn, independent of transport |
| `services/core/` | Stateless business logic (camera, monitor) |
| `core/`          | Config (the sole env parser), logging, errors |
| `app/websocket/` | Socket.IO transport binding only            |
| `app/api/v1/`    | REST endpoints                              |
