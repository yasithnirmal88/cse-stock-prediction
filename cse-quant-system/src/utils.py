"""Shared project utilities."""

import logging
from typing import Any


def configure_logging(config: dict[str, Any]) -> None:
    """Configure console logging from the project's logging settings."""
    settings = config.get("logging", {})
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid logging level: {level_name}")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
