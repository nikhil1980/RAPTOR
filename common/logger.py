import logging
import logging.handlers
import os
import sys
from typing import Optional
""" System Modules """

""" User Modules"""

"""
Script for setting up logging configuration across the application.

@author: Nikhil Bhargava
@date: 2024-06-15
@license: Apache-2.0
@description: This module configures logging settings for consistent logging behavior.
"""


# Take Defaults from os.getenv
_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
_DEFAULT_LOGFILE = os.getenv("LOG_FILE", "")  # empty = no file and send to console
_MAX_BYTES = int(os.getenv("LOG_ROTATE_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
_BACKUPS = int(os.getenv("LOG_ROTATE_BACKUPS", "5"))
_FMT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATEFMT = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

_configured = False  # internal guard to avoid duplicate handlers


def _configure_root_logger():
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(_DEFAULT_LEVEL)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(_DEFAULT_LEVEL)
    sh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(sh)

    # Optional rotating file handler
    if _DEFAULT_LOGFILE:
        fh = logging.handlers.RotatingFileHandler(
            _DEFAULT_LOGFILE, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
        fh.setLevel(_DEFAULT_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root.addHandler(fh)

    # Reduce noise from common libraries (tune as needed)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Usage:
        from common.logger import get_logger
        logger = get_logger(__name__)
        logger.info("hello")
    """
    _configure_root_logger()
    return logging.getLogger(name if name else "raptor")


def set_level(level: str):
    """Dynamically change level at runtime, e.g., set_level("DEBUG")."""
    _configure_root_logger()
    logging.getLogger().setLevel(level.upper())


def patch_print(logger: Optional[logging.Logger] = None):
    """
    (Optional) Redirect built-in print(...) to logger.info(...).
    Use only if you really want a drop-in replacement.
    """
    import builtins
    lg = logger or get_logger("print")
    def _print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "")
        msg = sep.join(str(a) for a in args) + end
        lg.info(msg)
    builtins.print = _print
