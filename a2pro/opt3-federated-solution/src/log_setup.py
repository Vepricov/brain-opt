"""Логирование по контракту platform.log/v1 через PlatformAPI.utils, с фолбэком."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> logging.Logger:
    """Настроить корневой логгер: JSON platform.log/v1 если есть PlatformAPI, иначе текст."""
    logger = logging.getLogger("a2pro_quant")
    logger.setLevel(logging.INFO)
    try:
        from PlatformAPI.utils import setup_logger

        setup_logger(logger)
    except Exception:  # noqa: BLE001 — офлайн / нет PlatformAPI
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
