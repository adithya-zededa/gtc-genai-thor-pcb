# Deployment Guide

## Docker

### Build

```bash
docker build -t camera-agent .
```

### Run

```bash
docker run -d \
  --name camera-agent \
  -p 8080:8080 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  camera-agent
```

### Docker Compose

```bash
docker-compose up -d
```

See [docker-compose.yml](../../docker-compose.yml) for the full service
definition, including the optional vLLM sidecar.

## Kubernetes (Helm)

A Helm chart is provided in `helm/camera-agent/`.

```bash
helm install camera-agent ./helm/camera-agent \
  --set image.tag=v57 \
  --set vllmServer.enabled=true
```

Key values:

| Value                 | Default       | Description                     |
|---------------------- |-------------- |-------------------------------- |
| `image.repository`    | `adithyazededa/gtc-genai-thor-pcb`| Container image |
| `image.tag`           | `v57`         | Image tag                       |
| `vllmServer.enabled`  | `true`        | Deploy vLLM server alongside camera-agent |
| `vllmServer.model`    | `nvidia/Cosmos-Reason2-8B` | VLM model name  |
| `ingress.enabled`     | `false`       | Expose via Ingress              |

See [helm/camera-agent/values.yaml](../../helm/camera-agent/values.yaml) for
all configurable values.

## Environment Variables

The application reads configuration from `config.yaml` and can be overridden
via environment variables.  See `config/settings.py` for the full list of
Pydantic settings fields.
