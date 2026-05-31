"""Project-wide logging setup.

A single ``configure_logging`` call gives every module consistent, timestamped
output. Library code uses ``logging.getLogger(__name__)`` and never configures
handlers itself, so importing the package stays side-effect free.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Idempotently configure root logging for scripts and apps."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured at least once."""
    configure_logging()
    return logging.getLogger(name)
