# ZEDEDA AI Agent Service

The ZEDEDA AI Agent is now a **pure Flask JSON service** that exposes the security agent and Ollama-backed language/vision models over HTTP. The browser-based frontend has been removed to simplify deployments and avoid client-side dependencies. All interaction happens through REST endpoints that can be called from scripts, services, or other applications.

## Overview

- ✅ No HTML or JavaScript is served — the application is API-only
- ✅ Consistent JSON responses designed for automation and integration
- ✅ Built-in health and diagnostics endpoints for operations teams
- ✅ Compatible with text-only or multi-modal (vision) prompts supported by Ollama

## Configuration

Set the following environment variables (defaults shown):

| Variable       | Default                | Description                                  |
| -------------- | ---------------------- | -------------------------------------------- |
| `OLLAMA_URL`   | `http://ollama:11434`  | Base URL for the Ollama server               |
| `OLLAMA_MODEL` | `llava:7b`             | Default model used when none is provided     |
| `PORT`         | `5000`                 | HTTP port exposed by the Flask application   |

The agent automatically loads `.env` configuration if present.

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.1"
python run.py
```

Once started, the service listens on `http://0.0.0.0:5000`.

## Container Image

```bash
docker build -f Dockerfile.web -t <your_tag> .
docker run --rm -p 5000:5000 -e OLLAMA_URL=http://host.docker.internal:11434 <your_tag>
```

## API Endpoints

| Method | Path                | Description                                                     |
| ------ | ------------------- | --------------------------------------------------------------- |
| `GET`  | `/`                 | Service metadata and quick reference                            |
| `GET`  | `/health`           | Basic health check (verifies connectivity to Ollama)            |
| `GET`  | `/api/version`      | Proxy to `OLLAMA_URL/api/version`                               |
| `POST` | `/api/generate`     | Execute a text or multi-modal generation request against Ollama |
| `POST` | `/api/agent/run`    | Run the higher-level security agent workflow                    |
| `GET`  | `/api/system-health`| Memory, disk, and Ollama connectivity diagnostics               |

### `/api/generate`

Request body:

```json
{
   "prompt": "Summarise this log",
   "model": "llama3.1",            // optional
   "stream": false,                 // optional
   "images": ["<base64>"]          // optional, pass-through to Ollama
}
```

Response mirrors Ollama's `generate` output and always includes the model used.

### `/api/agent/run`

Request body is passed directly to the autonomous agent:

```json
{
   "subject": "Door opened",
   "body": "Badge 9981 used at 03:42",
   "context": {"location": "HQ"}
}
```

Response:

```json
{
   "output": "Escalation not required.",
   "raw": { ... full agent response ... }
}
```

## Health & Diagnostics

- `GET /health` returns `200` when Ollama is reachable; otherwise `502`
- `GET /api/system-health` reports aggregated component status (`ollama`, `memory`, `disk`) with an overall state

## Helm & Kubernetes

The Kubernetes Helm chart under `helm/zededa-ai-agent` deploys the Flask service alongside an Ollama instance. Update `values.yaml` to point to the desired container tag and resource requirements before running:

```bash
helm upgrade --install email-agent ./helm/zededa-ai-agent
```

## Development Notes

- The Flask app is production-ready by running under Gunicorn (`Dockerfile.web`)
- All logging goes through Python's standard logging module (`LOG_LEVEL` respected)
- Integration tests can call the API endpoints directly; there is no browser dependency anymore

## License

Refer to the ZEDEDA licensing terms included with this repository.
