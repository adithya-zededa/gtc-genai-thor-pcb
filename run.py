"""Minimal Flask application exposing the Zededa AI agent over HTTP."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from email_agent.agent import AgentResponse, build_agent

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@dataclass
class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    base_url: str
    default_model: str
    request_timeout: int = 120

    def _full_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def version(self) -> Dict[str, Any]:
        response = requests.get(self._full_url("/api/version"), timeout=10)
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        stream: bool = False,
        images: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": stream,
        }
        if images:
            payload["images"] = images
        if options:
            payload["options"] = options

        response = requests.post(
            self._full_url("/api/generate"),
            json=payload,
            timeout=timeout or self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        data.setdefault("model", payload["model"])
        return data

    def ping(self) -> bool:
        try:
            self.version()
        except requests.RequestException as exc:
            logger.warning("Failed to reach Ollama: %s", exc)
            return False
        return True


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_BASE_URL = OLLAMA_URL.rstrip("/")

ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL, default_model=OLLAMA_MODEL)
agent = build_agent(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

app = Flask(__name__)


@app.before_request
def log_request_info() -> None:
    logger.info("Request: %s %s", request.method, request.url)
    if request.is_json:
        payload = request.get_json(silent=True)
        logger.info("Request JSON keys: %s", list(payload.keys()) if isinstance(payload, dict) else payload)


@app.after_request
def log_response_info(response):  # type: ignore[override]
    logger.info("Response: %s %s", response.status_code, response.status)
    return response


@app.get("/")
def service_description() -> Any:
    return jsonify(
        {
            "service": "zededa-ai-agent",
            "status": "ready",
            "model": OLLAMA_MODEL,
            "ollama_url": OLLAMA_URL,
            "endpoints": {
                "GET /health": "Basic status check",
                "GET /api/version": "Proxy Ollama version information",
                "POST /api/generate": "Run a text or vision generation request",
                "POST /api/agent/run": "Execute the higher-level agent workflow",
                "GET /api/system-health": "Infrastructure health snapshot",
            },
        }
    )


@app.get("/health")
def health() -> Any:
    status = "healthy" if ollama_client.ping() else "degraded"
    return jsonify({
        "status": status,
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
    }), (200 if status == "healthy" else 502)


def _proxy_error(message: str, detail: str) -> Any:
    logger.warning("Ollama proxy error: %s (%s)", message, detail)
    return jsonify({"error": message, "detail": detail}), 502


@app.get("/api/version")
def proxy_version() -> Any:
    try:
        data = ollama_client.version()
    except requests.RequestException as exc:  # pragma: no cover - network path
        return _proxy_error("Failed to reach Ollama /api/version", str(exc))
    return jsonify(data)


@app.post("/api/generate")
def proxy_generate() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        return jsonify({"error": "prompt must be a string"}), 400

    images = payload.get("images")
    if images is not None and not isinstance(images, list):
        return jsonify({"error": "images must be a list of base64 strings"}), 400

    options = payload.get("options")
    if options is not None and not isinstance(options, dict):
        return jsonify({"error": "options must be an object"}), 400

    timeout = payload.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        return jsonify({"error": "timeout must be a positive number"}), 400

    try:
        data = ollama_client.generate(
            prompt=prompt,
            model=payload.get("model"),
            stream=bool(payload.get("stream", False)),
            images=images,
            options=options,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - network path
        return _proxy_error("Failed to reach Ollama /api/generate", str(exc))

    return jsonify(data)


@app.get("/api/system-health")
def system_health() -> Any:
    """Aggregate infrastructure health data for operators."""

    try:
        version_info = ollama_client.version()
        ollama_status = {
            "status": "ok",
            "detail": version_info.get("version", "unknown version"),
        }
    except requests.RequestException as exc:
        ollama_status = {
            "status": "error",
            "detail": str(exc),
        }

    virtual_mem = psutil.virtual_memory()
    memory_pct = virtual_mem.percent
    memory_status = {
        "status": "ok" if memory_pct < 90 else "warning",
        "detail": f"{memory_pct:.1f}% used",
    }

    disk_usage = shutil.disk_usage("/")
    disk_pct = (disk_usage.used / disk_usage.total) * 100
    disk_status = {
        "status": "ok" if disk_pct < 90 else "warning",
        "detail": f"{disk_pct:.1f}% used",
    }

    component_statuses = {
        "ollama": ollama_status,
        "memory": memory_status,
        "disk": disk_status,
    }

    overall = "ok"
    for component in component_statuses.values():
        if component["status"] == "error":
            overall = "error"
            break
        if component["status"] == "warning" and overall != "error":
            overall = "warning"

    payload = {
        "overall": overall,
        "components": component_statuses,
        "last_checked": datetime.utcnow().isoformat() + "Z",
    }

    return jsonify(payload)


@app.post("/api/agent/run")
def run_agent() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    context = payload.pop("context", None)

    logger.info("Received agent request with keys: %s", list(payload.keys()))

    response: AgentResponse = agent.run(payload=payload, context=context)
    result = response.to_dict()

    return jsonify({
        "output": response.output_text,
        "raw": result,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
