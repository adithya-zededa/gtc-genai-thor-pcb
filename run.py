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
    
    logger.info("=" * 60)
    logger.info("ZEDEDA Camera Monitoring Agent")
    logger.info("=" * 60)
    logger.info("Configuration loaded from: %s", os.getenv("CAMERA_AGENT_CONFIG", "config.yaml"))
    logger.info("Flask host: %s, port: %d", config.flask.host, config.flask.port)
    logger.info("Inference backend: %s", config.inference.backend)
    logger.info("Vision model: %s", config.inference.vision_model)
    
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
