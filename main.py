"""
Entry point — initialises storage, then launches the Gradio app.

Usage:
    python main.py                  # production
    DEBUG=true python main.py       # verbose logging
"""

import asyncio
import sys

from core.config import settings
from core.logger import get_logger
from storage.db import init_db
from ui.app import create_app

logger = get_logger(__name__)


async def startup() -> None:
    """Run all async startup tasks before the UI launches."""
    settings.ensure_dirs()
    await init_db()
    logger.info(
        "Starting Meeting Summariser  |  model=%s  |  whisper=%s  |  device=%s",
        settings.ollama_llm_model,
        settings.whisper_model_size,
        settings.whisper_device,
    )


def main() -> None:
    # Run startup checks
    try:
        asyncio.run(startup())
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        sys.exit(1)

    app = create_app()
    app.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        debug=settings.debug,
        share=False,
        theme="soft",
    )


if __name__ == "__main__":
    main()