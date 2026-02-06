# Development Guide

## Prerequisites

- Python 3.10+
- (Optional) Docker & Docker Compose for containerised runs
- (Optional) NVIDIA GPU + CUDA for local vLLM inference

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd agent-attempt-2

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dev dependencies (includes testing & linting)
pip install -r requirements/dev.txt

# Copy and edit configuration
cp config.yaml config.local.yaml
# Edit config.local.yaml with your API keys / camera settings

# Run the development server
python run.py
```

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
| `agents/core/`   | Core agent loop, state, alerting           |
| `agents/tools/`  | Callable tools (each file = one domain)    |
| `services/core/` | Stateless business logic (camera, monitor) |
| `config/`        | Pydantic models, defaults, schemas         |
| `core/`          | Cross-cutting utilities & error classes    |
