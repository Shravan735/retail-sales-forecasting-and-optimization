"""Compatibility entrypoint for older commands.

The maintained FastAPI implementation lives in ``backend.server``. Keeping this
thin module avoids two backend copies drifting apart while preserving
``uvicorn backend.main:app`` for anyone who used the older entrypoint.
"""

from backend.server import app, chat_with_run, health_check, process_dataset, upload_dataset

__all__ = ["app", "chat_with_run", "health_check", "process_dataset", "upload_dataset"]
