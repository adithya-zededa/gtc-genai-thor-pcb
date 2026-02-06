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
  -p 5005:5005 \
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
  --set image.tag=latest \
  --set vllm.enabled=true
```

Key values:

| Value                 | Default       | Description                     |
|---------------------- |-------------- |-------------------------------- |
| `image.repository`    | `camera-agent`| Container image                 |
| `image.tag`           | `latest`      | Image tag                       |
| `vllm.enabled`        | `false`       | Deploy vLLM sidecar             |
| `vllm.model`          | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM model name  |
| `ingress.enabled`     | `false`       | Expose via Ingress              |

See [helm/camera-agent/values.yaml](../../helm/camera-agent/values.yaml) for
all configurable values.

## Environment Variables

The application reads configuration from `config.yaml` and can be overridden
via environment variables.  See `config/settings.py` for the full list of
Pydantic settings fields.
