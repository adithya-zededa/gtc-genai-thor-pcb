"""WSGI entry point for production deployment.

Usage with gunicorn:
    gunicorn --worker-class eventlet -w 1 wsgi:application

Usage with waitress (Windows compatible):
    from waitress import serve
    serve(application, host='0.0.0.0', port=8080)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Setup logging
from core.logging import setup_logging
setup_logging()

# Initialize database
from app.database import init_db, ensure_database_directory, start_retention_worker
ensure_database_directory()
init_db()

# Bound the append-only datasets (see app/database/maintenance.py)
start_retention_worker()

# Create application
from app import create_app, socketio

application = create_app()

# For gunicorn with eventlet/gevent
app = application

if __name__ == "__main__":
    # Direct WSGI run (for testing)
    from core.config import get_config
    
    config = get_config()
    socketio.run(
        application,
        host=config.flask.host,
        port=config.flask.port,
    )
