"""Application logging configuration."""

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure concise stderr logging for CLI workflows."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
