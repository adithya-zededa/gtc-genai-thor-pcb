#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Main Entry Point

This script starts the Flask web application with the camera monitoring agent.

Initialization order:
1. Environment variables (dotenv)
2. Configuration loading
3. Logging setup
4. Database initialization
5. Flask app creation (includes blueprint registration)
6. SocketIO server start
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


def main():
    """Main entry point for running the application."""
    # 1. Import and setup logging (after env vars are loaded)
    from core.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    
    # 2. Load configuration
    from core.config import get_config
    config = get_config()
    
    # 2a. Auto-detect the served models if not explicitly set
    from core.model_detect import detect_model
    detected_vision = detect_model(
        backend="vllm",
        base_url=config.inference.vllm_url,
        env_override="VISION_MODEL",
    )
    if not config.inference.model:
        config.inference.model = detected_vision

    detected_agent = detect_model(
        backend="vllm",
        base_url=config.router.url,
        env_override="AGENT_MODEL",
    )
    if not config.router.model:
        config.router.model = detected_agent

    logger.info("=" * 60)
    logger.info("ZEDEDA Camera Monitoring Agent")
    logger.info("=" * 60)
    logger.info("Configuration loaded from: %s", os.getenv("CAMERA_AGENT_CONFIG", "config.yaml"))
    logger.info("Flask host: %s, port: %d", config.flask.host, config.flask.port)
    logger.info("Inference backend: vLLM")
    logger.info("Vision model: %s (%s)", config.inference.model, config.inference.vllm_url)
    logger.info("Agent model: %s (%s)", config.router.model, config.router.url)

    # 2b. Initialize LLM routers (agent + vision are independent instances)
    try:
        from router import get_router
        agent_router = get_router(role="agent")
        agent_router.configure(model=config.router.model)
        vision_router = get_router(role="vision")
        vision_router.configure(model=config.inference.model)

        logger.info("LLM Router enabled – agent + vision providers configured")
        for label, r in (("agent", agent_router), ("vision", vision_router)):
            for p in r.list_providers():
                status = "✅" if p.get("status", {}).get("available") else "❌"
                logger.info("  %s [%s] %s model=%s", status, label, p["name"], p.get("model", "auto"))
    except Exception as exc:
        logger.warning("LLM Router failed to initialize: %s", exc)
    
    # 3. Initialize database
    from app.database import init_db, ensure_database_directory
    ensure_database_directory()
    init_db()
    logger.info("Database initialized: %s", config.database.path)
    
    # 4. Create Flask application (registers blueprints and websocket handlers)
    from app import create_app, socketio
    app = create_app()
    logger.info("Flask application created")
    
    # 5. Run with Socket.IO
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    
    logger.info("Starting server (debug=%s)...", debug)
    logger.info("=" * 60)
    
    socketio.run(
        app,
        host=config.flask.host,
        port=config.flask.port,
        debug=debug,
        use_reloader=debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
