# Architecture Overview

## High-Level Design

The ZEDEDA Camera Monitoring Agent is a Flask-based application that provides
real-time camera monitoring with AI-powered inference capabilities.

## Directory Structure

```
.
├── agents/              # Agent logic
│   ├── core/            # Monitoring loop, detection agent, state
│   ├── tools/           # Tool definitions & executors (email, PCB)
│   ├── mcp/             # Model Context Protocol server & domain MCPs
│   ├── classifiers/     # LLM-based classification
│   └── vlm/             # Vision Language Model client & prompts
├── app/                 # Flask application
│   ├── api/             # REST API (versioned under v1/, v2/, …)
│   ├── database/        # SQLAlchemy-style models & repositories
│   ├── views/           # Jinja2 HTML views
│   └── websocket/       # Socket.IO real-time handlers
├── config/              # Pydantic settings, YAML defaults, schemas
├── core/                # Cross-cutting: config, logging, errors, utils
├── router/              # LLM Router with multi-provider failover
│   └── adapters/        # Provider adapters (OpenAI, Anthropic, Google, …)
├── services/            # Business logic
│   ├── core/            # Camera, monitoring, inference services
│   ├── infrastructure/  # Config I/O, VLM client factory
│   └── domains/         # Domain-specific (PCB)
├── requirements/        # Split dependency manifests
├── tests/               # Test suite
│   ├── unit/            # Fast isolated tests
│   ├── integration/     # Cross-module tests
│   ├── e2e/             # End-to-end tests
│   └── fixtures/        # Shared test fixtures & mocks
├── helm/                # Kubernetes Helm chart
├── templates/           # Jinja2 HTML templates
└── static/              # CSS, JS, images
```

## Key Design Patterns

### Application Factory
The Flask app is created via `create_app()` in `app/__init__.py`, enabling
multiple configurations (production, testing, development).

### Repository Pattern
Database access is encapsulated in repository classes under
`app/database/repositories.py`, backed by `BaseRepository[T]` in
`app/database/base_repository.py`.

### MCP (Model Context Protocol)
Tool calling follows a propose → approve → execute lifecycle managed by
`agents/mcp/base.py` and dispatched by `agents/mcp/manager.py`.

### LLM Router
Multi-provider LLM routing with automatic failover, rate limiting, and
circuit breakers lives in `router/`.

### Error Hierarchy
All application errors inherit from `CameraAgentError` in `core/errors.py`,
carrying an HTTP `status_code` and structured `details` dict.
