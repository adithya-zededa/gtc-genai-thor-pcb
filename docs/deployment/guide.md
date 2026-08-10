# Deployment Guide

## Docker

### Build

```bash
docker build -t camera-agent .
```

### Run

`VLLM_URL` is **required** — `start.sh` exits immediately without it, since
the image carries no inference engine of its own.

```bash
docker run -d \
  --name camera-agent \
  -p 8080:8080 \
  -e VLLM_URL=http://host.docker.internal:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  camera-agent
```

To serve the agent (text) model from a second endpoint, add
`-e AGENT_LLM_URL=... -e AGENT_MODEL=...`. When unset, both roles share
`VLLM_URL`.

### Docker Compose

```bash
docker-compose up -d
```

See [docker-compose.yml](../../docker-compose.yml) for the full service
definition. The vLLM service is **required**, not optional: `camera-agent`
declares `depends_on: vllm-server` and refuses to start without a reachable
`VLLM_URL`. Compose runs a single vLLM (`Qwen/Qwen3-VL-4B-Instruct`) serving
both roles; the Helm chart splits them across two pods.

## Kubernetes (Helm)

A Helm chart is provided in `helm/camera-agent/`.

```bash
helm install camera-agent ./helm/camera-agent
```

The chart deploys three workloads by default: the app, a vision vLLM, and an
agent vLLM. The two model pods share one GPU via device-plugin time slicing,
so the node must advertise at least 2 `nvidia.com/gpu` replicas:

```bash
kubectl get node -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}'
```

If it reports `1`, enable `gpuTimeSlicing` (or configure the device plugin
out-of-band) or the second pod stays `Pending`. To run a single model
instead, set `vllmAgent.enabled=false` — both roles then share the vision pod.

Key values:

| Value                 | Default       | Description                     |
|---------------------- |-------------- |-------------------------------- |
| `image.repository`    | `adithyazededa/gtc-genai-thor-pcb`| Container image |
| `image.tag`           | `v57`         | Image tag                       |
| `vllmServer.enabled`  | `true`        | Deploy the vision vLLM server   |
| `vllmServer.model`    | `LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect` | Vision model |
| `vllmAgent.enabled`   | `true`        | Deploy the agent (text) vLLM server |
| `vllmAgent.model`     | `LiquidAI/LFM2.5-2.6B` | Agent model            |
| `gpuTimeSlicing.enabled` | `false`    | Let the device plugin advertise 2 GPU replicas |
| `ingress.enabled`     | `false`       | Expose via Ingress              |
| `replicaCount`        | `1`           | **Must stay 1** — see the architecture doc |

See [helm/camera-agent/values.yaml](../../helm/camera-agent/values.yaml) for
all configurable values.

## Environment Variables

Precedence is **environment > `config.yaml` > default**, and the environment
layer is parsed in exactly one place: [`core/config.py`](../../core/config.py).
That module's `Config` dataclass tree is the full list of settings.

Frequently used:

| Variable | Purpose |
|---|---|
| `VLLM_URL` | Vision model endpoint (**required**) |
| `VISION_MODEL` | Vision model id; auto-detected from the server when unset |
| `AGENT_LLM_URL` / `AGENT_MODEL` | Agent model endpoint; falls back to the vision pod |
| `CAMERA_AGENT_DATA_DIR` | Root for the database, captured images, and the secret key |
| `CAMERA_AGENT_CONFIG` | Path to the YAML config (default `config.yaml`) |
| `RETENTION_INTERVAL_SECONDS` | How often the retention sweep runs (default 3600) |
| `EMAIL_SMTP_SERVER` / `EMAIL_USER` / `EMAIL_PASS` | SMTP; overrides the `notifications.email` block in `config.yaml` |

## Probes

The chart wires each Kubernetes probe to a purpose-built endpoint — see
[Health and probes](../architecture/README.md#health-and-probes). Do not point
them at `/api/status`: it returns 200 unconditionally and cannot fail.
