"""Phase 1 entry point: load and report project configuration only."""

import logging
from typing import Sequence

from src.data_loader import load_config, resolve_data_path
from src.utils import configure_logging


def main(args: Sequence[str] | None = None) -> int:
    """Load configuration and report configured locations without fetching data."""
    del args  # Reserved for future command-line options.
    config = load_config()
    configure_logging(config)
    logger = logging.getLogger(__name__)
    processed_dir = resolve_data_path(config, "processed_dir")
    logger.info("Loaded %s version %s", config["project"]["name"], config["project"]["version"])
    logger.info("Processed data directory: %s", processed_dir)
    logger.info("Phase 1 performs no data downloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
